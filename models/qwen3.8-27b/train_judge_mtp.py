"""
train_judge_mtp.py -- align the MTP drafter to the trunk that actually ships.

ONE GPU, a plain loop, no FSDP and no Trainer. That is a measured decision, not
a shortcut: the trunk is frozen, so only the 424.7M-parameter head carries
gradients and optimizer state, and the head probe peaked at 106 GiB on a single
B300 with two rows in flight. Sharding a frozen trunk across eight GPUs to train
1.5% of the model would add a distributed failure surface to buy wall-clock on a
run that takes about a quarter of an hour.

WHY THIS EXISTS

rc0 shipped the BASE drafter, copied verbatim, because transformers has no MTP
implementation for this architecture and no adapter can attach to a module that
is never instantiated. Measured cost of that transplant: the trunk's final
hidden state moved to cosine 0.788 after the merge, and acceptance length fell
to 2.93 against 4.39 for a matched head -- roughly 7.5 tok/s against ~11. It
costs throughput and cannot cost quality, since verification is done by the
target model.

THE SEQUENCING, WHICH IS THE PART THAT IS EASY TO GET WRONG

A drafter is only aligned to one trunk. So for rc1 the order is:

    1. train the LoRA           -> adapter
    2. merge it                 -> the trunk that ships
    3. train the head HERE, against that merged trunk
    4. merge_judge_lora.py --mtp-head <this output>, to write it in

Training the head against the BASE trunk would reproduce exactly the problem it
is meant to solve. MODEL_PATH therefore defaults to a merged checkpoint and the
script says so loudly if it is handed the base.

The head is initialised from whatever `mtp.*` tensors the given checkpoint
carries -- which, for a checkpoint merged by merge_judge_lora.py, are the base
head's. Starting from a trained drafter and adapting it is cheaper than starting
from noise, and it is the only initialisation available.
"""

import json
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from model import load_vision_model
from mtp import MTPTrainer, Qwen3_5MTPHead, load_base_mtp_weights, verify_alignment
from tasks.judge import dataset as judge_dataset
from tasks.judge.collator import JudgeCollator, build_feature
from tasks.judge.masking import IGNORE_INDEX, assert_masking_is_sane


def save_head(head, output_dir):
    """
    Write the head as `mtp.`-prefixed tensors, ready to install into a checkpoint.

    The prefix is restored on the way out so the file is a drop-in for the 15
    tensors a checkpoint carries -- merge_judge_lora.py --mtp-head reads exactly
    this. Saving it unprefixed would produce a file that looks installable and
    silently matches nothing.
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    state = {f"mtp.{k}": v.detach().cpu().contiguous()
             for k, v in head.state_dict().items()}
    path = os.path.join(output_dir, "mtp_head.safetensors")
    save_file(state, path, metadata={"format": "pt"})
    return path, state


def main():
    templates = (
        os.environ.get("TEMPLATES_DIR")
        or os.environ.get("DEMO_DIR")
        or "/mnt/data/slurm-llm-templates"
    )
    base = f"{templates}/models/Qwen3.8-27B"
    model_path = os.environ.get(
        "MODEL_PATH", f"{templates}/output/qwen3.8-27b-judge-rc0-merged-complete"
    )
    dataset_dir = os.environ.get(
        "JUDGE_DATASET_DIR", f"{templates}/datasets/judge-sft-rc0-54df8b59"
    )
    output_dir = os.environ.get("OUTPUT_DIR", f"{templates}/output/qwen3.8-27b-judge-mtp")
    epochs = int(os.environ.get("NUM_EPOCHS", "3"))
    lr = float(os.environ.get("LEARNING_RATE", "1e-4"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "10240"))
    n_eval = int(os.environ.get("N_EVAL_ROWS", str(judge_dataset.DEFAULT_N_EVAL)))
    warmup_frac = float(os.environ.get("WARMUP_RATIO", "0.1"))

    print(f"Trunk      : {model_path}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Output     : {output_dir}")
    print(f"Epochs     : {epochs}   LR: {lr}   max_seq_len: {max_seq_len}")

    if os.path.realpath(model_path) == os.path.realpath(base):
        print("\n  WARNING: training the head against the BASE trunk.")
        print("  A drafter is aligned to exactly one trunk, and the base is not")
        print("  what ships. This reproduces the problem the head exists to fix.")
        print("  Set MODEL_PATH to the MERGED checkpoint unless you are probing.\n")

    rows = judge_dataset.load_rows(dataset_dir)
    train_rows, eval_rows = judge_dataset.train_eval_split(rows, dataset_dir, n_eval)
    if os.environ.get("DROP_AUTO_C", "0") not in ("0", "false", "False"):
        train_rows, dropped = judge_dataset.drop_auto_c_rows(train_rows, dataset_dir)
        print(f"DROP_AUTO_C : dropped {len(dropped)} auto-C-only rows; {len(train_rows)} train rows remain")
    # MAX_ROWS caps the training rows for a smoke run. Deliberately AFTER the
    # split, so a smoke run exercises the same carve the real run will use
    # rather than a different one -- the repo's standing rule that short runs
    # find defects, not numbers, only holds if they run the same code path.
    max_rows = int(os.environ.get("MAX_ROWS", "-1"))
    if 0 <= max_rows < len(train_rows):
        train_rows = train_rows[:max_rows]
        print(f"MAX_ROWS   : {max_rows} -- SMOKE RUN, not a usable head")
    print(f"Rows       : {len(train_rows)} train + {len(eval_rows)} drift guard")

    processor = judge_dataset.load_processor(model_path)
    t0 = time.perf_counter()
    assert_masking_is_sane(processor, train_rows, dataset_dir)
    print(f"Mask gate  : {len(train_rows)}/{len(train_rows)} pass "
          f"({time.perf_counter() - t0:.0f}s)")

    collator = JudgeCollator(pad_token_id=processor.tokenizer.pad_token_id,
                             max_seq_len=max_seq_len)

    model = load_vision_model(model_path)
    head = Qwen3_5MTPHead(model.config.text_config).to(dtype=torch.bfloat16)
    loaded = load_base_mtp_weights(head, model_path, strict=True)
    print(f"Head       : {sum(p.numel() for p in head.parameters()):,} params, "
          f"{len(loaded)} tensors initialised from the checkpoint")

    model.cuda()
    head.cuda()
    trainer = MTPTrainer(model, head, train_trunk=False)

    # The alignment check from the probe, on this trunk and this data, before a
    # single optimizer step. An off-by-one trains to a plausible loss.
    probe_batch = collator([build_feature(processor, train_rows[0], dataset_dir)])
    probe_batch = {k: v.cuda() for k, v in probe_batch.items()}
    trainer.eval()
    res = verify_alignment(trainer, probe_batch, model.config.text_config.vocab_size)
    chance = res.pop("chance")
    print(f"Alignment  : +1 {res[1]:.3f} | +2 {res[2]:.3f} | +3 {res[3]:.3f} "
          f"| chance {chance:.3f}")
    if min(res, key=res.get) != 2 or res[2] >= chance / 2:
        raise SystemExit("alignment check failed; refusing to train on this objective")

    trainable = [p for p in trainer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_rows)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=warmup_frac, anneal_strategy="cos",
    )

    def loss_over(rows_subset):
        trainer.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for row in rows_subset:
                b = collator([build_feature(processor, row, dataset_dir)])
                out = trainer(**{k: v.cuda() for k, v in b.items()})
                total += float(out["loss"])
                n += 1
        trainer.train()
        return total / max(n, 1)

    print(f"\nTraining {total_steps} steps ({epochs} epochs x {len(train_rows)} rows)")
    torch.cuda.reset_peak_memory_stats()
    history, step, t_start = [], 0, time.perf_counter()
    trainer.train()
    for epoch in range(1, epochs + 1):
        running, seen = 0.0, 0
        for row in train_rows:
            batch = collator([build_feature(processor, row, dataset_dir)])
            out = trainer(**{k: v.cuda() for k, v in batch.items()})
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            sched.step()
            optimizer.zero_grad(set_to_none=True)
            running += out["loss"].detach().item()
            seen += 1
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{total_steps}  train {running / seen:.4f}  "
                      f"{(time.perf_counter() - t_start) / step:.2f} s/step")
        guard = loss_over(eval_rows) if eval_rows else None
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "drift_guard": guard})
        print(f"  epoch {epoch}: train {history[-1]['train_loss']:.4f}"
              + (f"  drift guard {guard:.4f}" if guard is not None else ""))

    seconds = time.perf_counter() - t_start
    path, state = save_head(head, output_dir)
    record = {
        "trunk": model_path,
        "dataset_dir": dataset_dir,
        "rows_train": len(train_rows),
        "rows_drift_guard": len(eval_rows),
        "drift_guard_ids": [r["id"] for r in eval_rows],
        "epochs": epochs, "lr": lr, "warmup_ratio": warmup_frac,
        "max_seq_len": max_seq_len,
        "head_params": sum(p.numel() for p in head.parameters()),
        "tensors": sorted(state),
        "alignment_losses": {str(k): v for k, v in res.items()} | {"chance": chance},
        "history": history,
        "train_seconds": round(seconds, 1),
        "peak_gib_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_gib_reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    with open(os.path.join(output_dir, "mtp_recipe.json"), "w") as handle:
        json.dump(record, handle, indent=1)

    trainer.close()
    print(f"\nDone in {seconds / 60:.1f} min. Head -> {path}")
    print(f"Peak {record['peak_gib_allocated']:.2f} GiB allocated, "
          f"{record['peak_gib_reserved']:.2f} GiB reserved")
    print("\nInstall it into the shipping checkpoint with:")
    print(f"  merge_judge_lora.py --mtp-head {path} <adapter> <base> <output>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
