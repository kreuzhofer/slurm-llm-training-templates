"""
judge_step_probe.py -- the first real gradient step on judge data (#23).

One B300, one repeated batch, forward -> backward -> step. Not a trainer: a
measurement and a proof that data, mask, model and optimizer are connected.

WHY OVERFIT-ONE-BATCH IS THE TEST

It is the cheapest proof that the pipeline learns at all. A pipeline that is
silently learning nothing -- labels all masked, adapters not attached, grads
not flowing through a checkpointed layer -- sits flat on a repeated batch,
while a working one drops fast because there is nothing to generalise to. A
mask bug that survived #22's checks would show up here as a loss that falls to
a floor that is not near zero, or does not fall at all.

WHAT IT REPLACES

A projection. The 2.0-2.5 s/step figure on the map is extrapolated from the
text-only SQL runs plus the -8.0% tower cost measured in #20 -- and #20 ran
with an IDLE tower: no image has ever been forward-passed in training on this
cluster. The map's own standing warning is that three projections extrapolated
from smoke runs were wrong by 2-5x, so this prints a measured number and says
which number it is.

Run via judge_step_probe.sbatch (one GPU, no torchrun, no FSDP -- the whole
27.36B model in bf16 is ~51 GiB against 268.6 GiB of HBM, so sharding would
only add a variable the measurement does not need).
"""

import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from peft import LoraConfig, TaskType, get_peft_model

from model import disable_kv_cache, load_vision_model
from tasks.judge import dataset as judge_dataset
from tasks.judge.collator import JudgeCollator, build_feature
from tasks.judge.masking import IGNORE_INDEX, assert_masking_is_sane

# Attention + MLP, no Gated DeltaNet projections.
#
# Duplicated from train_lora.lora_target_modules() rather than imported,
# because that module pulls in the SQL task at import time and a judge run has
# no business loading it. The reasoning and the measurement behind this list
# live there and are not restated: #12 ran both arms at two scales and the GDN
# projections did not win a single held-out example, at 58.2M extra trainable
# parameters and ~15% slower steps.
#
# DEBT: the list is architecture-shaped, not task-shaped, so it belongs in
# model.py. Moving it is part of the common/ -> tasks/sql/ rename (#18's
# decision 4), not of this ticket -- but two tasks reading two copies of this
# list is exactly the drift the repo keeps warning about, so it should not
# survive long.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def assert_vision_is_unadapted(model):
    """
    No LoRA adapter may sit on the vision tower.

    Recorded on chat3d #95 as a hazard of the sibling recipe: "suffix matching
    captures its q/k/v/o". Verified FALSE for this checkpoint -- its tower uses
    `qkv`, `proj`, `linear_fc1`, `linear_fc2`, so none of the seven names above
    can match it, and a bare suffix list captures exactly 0 vision modules.

    Checked anyway, and by inspecting the built model rather than by trusting
    that note: the naming is transformers' to change, and the failure it
    guards against -- a tower quietly being adapted, making the run neither
    the frozen-tower arm nor a stated alternative -- would not announce itself
    in the loss.
    """
    adapted = [
        n for n, _ in model.named_modules()
        if ("lora_A" in n or "lora_B" in n) and (".visual." in n or n.startswith("visual."))
    ]
    assert not adapted, (
        f"{len(adapted)} LoRA adapters landed on the vision tower, e.g. "
        f"{adapted[:3]}. The tower must stay frozen; the target list or the "
        "module naming changed."
    )
    trainable_vision = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (".visual." in n or n.startswith("visual."))
    ]
    assert not trainable_vision, (
        f"{len(trainable_vision)} vision parameters are trainable, e.g. "
        f"{trainable_vision[:3]}"
    )


def pick_mixed_batch(rows, per_device_bs):
    """
    A batch holding both render sizes, because the export holds both.

    Four of the 402 rows are all-512x512 (2,048 visual tokens against 4,608);
    a batch drawn from the head of the file would be uniformly 768 and would
    not exercise the concatenation the collator does. Different lengths are
    the point here, not a nuisance to be avoided.
    """
    small = [r for r in rows if len(r["messages"]) and r["id"].startswith("dac780aa")]
    if not small:
        # Fall back to whichever row is shortest; the size mix is a property of
        # the export, so a future one may not contain this id.
        small = [min(rows, key=lambda r: len(str(r["messages"])))]
    others = [r for r in rows if r["id"] != small[0]["id"]]
    return [small[0]] + others[: max(0, per_device_bs - 1)]


def main():
    templates_dir = (
        os.environ.get("TEMPLATES_DIR")
        or os.environ.get("DEMO_DIR")
        or "/mnt/data/slurm-llm-templates"
    )
    model_path = os.environ.get("MODEL_PATH", f"{templates_dir}/models/Qwen3.8-27B")
    dataset_dir = os.environ.get(
        "JUDGE_DATASET_DIR", f"{templates_dir}/datasets/judge-sft-rc0-54df8b59"
    )
    steps = int(os.environ.get("PROBE_STEPS", "10"))
    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "2"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "10240"))
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    grad_ckpt = os.environ.get("GRADIENT_CHECKPOINTING", "1") not in ("0", "false")

    print(f"Model       : {model_path}")
    print(f"Dataset     : {dataset_dir}")
    print(f"Batch       : {per_device_bs} rows/step, repeated {steps} times")
    print(f"LoRA        : r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Targets     : {LORA_TARGET_MODULES}")
    print(f"Max seq len : {max_seq_len}")
    print(f"Grad ckpt   : {'on' if grad_ckpt else 'off'}")
    print(f"LR          : {learning_rate}")

    rows = judge_dataset.load_rows(dataset_dir)
    processor = judge_dataset.load_processor(model_path)
    batch_rows = pick_mixed_batch(rows, per_device_bs)

    # The load-time gate from #22, on exactly the rows about to be trained.
    assert_masking_is_sane(processor, batch_rows, dataset_dir)
    print(f"\nMasking gate: {len(batch_rows)} rows pass")

    features = [build_feature(processor, r, dataset_dir) for r in batch_rows]
    for r, f in zip(batch_rows, features):
        n_img = int(f["image_grid_thw"].shape[0])
        sizes = sorted({f"{int(h) * 16}x{int(w) * 16}" for _, h, w in f["image_grid_thw"]})
        print(f"  {r['id'][:8]}  {f['input_ids'].shape[0]:>6} tokens, "
              f"{int((f['labels'] != IGNORE_INDEX).sum()):>4} supervised, "
              f"{n_img} images {sizes}")

    collator = JudgeCollator(
        pad_token_id=processor.tokenizer.pad_token_id, max_seq_len=max_seq_len
    )
    batch = collator(features)
    print("\nCollated:")
    for k, v in batch.items():
        print(f"  {k:22} {tuple(v.shape)}")

    model = load_vision_model(model_path)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        ),
    )
    assert_vision_is_unadapted(model)
    print()
    model.print_trainable_parameters()
    print("vision tower: 0 adapters, 0 trainable parameters (asserted)")

    if grad_ckpt:
        # use_reentrant=False is not a style preference here. Under the
        # reentrant implementation the checkpointed segment's inputs do not
        # require grad -- the base weights are frozen and only the adapters
        # are trainable -- so no gradient reaches the LoRA parameters and the
        # loss sits flat. Which is precisely the failure this probe exists to
        # detect, so it must not be self-inflicted.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    model.cuda()
    disable_kv_cache(model)  # again: get_peft_model wraps, and config is re-read
    model.train()

    batch = {k: v.cuda() for k, v in batch.items()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    torch.cuda.reset_peak_memory_stats()
    print(f"\nOverfitting one batch for {steps} steps:")
    losses, times = [], []
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        losses.append(loss.detach().item())
        times.append(dt)
        print(f"  step {step + 1:>2}  loss {losses[-1]:8.4f}  {dt:6.2f} s")

    assert all(l == l for l in losses), "loss went NaN"
    assert losses[0] != losses[-1], "loss did not move at all"

    # Steady-state excludes step 1, which carries lazy allocator growth and
    # kernel autotuning; the real run's per-step cost is the rest.
    steady = times[1:] or times
    print(f"\nloss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({100 * (losses[0] - losses[-1]) / max(losses[0], 1e-9):.1f}% down)")
    print(f"step time: {sum(steady) / len(steady):.2f} s mean over steps 2-{steps} "
          f"(first step {times[0]:.2f} s), {per_device_bs} rows/step")
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved")

    if losses[-1] >= losses[0]:
        print("\nFAIL: loss did not fall on a repeated batch.")
        return 1
    print("\nPASS: data, mask, model and optimizer are connected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
