"""
train_judge_lora.py -- LoRA fine-tune Qwen3.8-27B on the rc0 judge task (#24).

Run via train_judge_lora.sbatch (torchrun, one node x 8 B300).

A SECOND TRAINING SCRIPT, NOT A FLAG ON train_lora.py

Same reasoning as load_model / load_vision_model in model.py: the SQL numbers
in docs/RESULTS.md came out of train_lora.py and have to stay reproducible, so
that path keeps exactly the shape it had. This one loads the vision tower,
builds its own labels, and collates image tensors that the SQL path has no
concept of.

WHAT THE NUMBERS HERE ARE, AND ARE NOT

This run publishes NO accuracy. rc0's real evaluation is the qualification
screen on the held-out 125, which lives on chat3d's side, is disjoint from
this export by example id and by prompt, and carries human adjudication on the
disagreements. What this script puts on the record is training health:
per-epoch train loss, a preservation loss on ten agreed-only rows, step time,
and peak memory.

The eval loss is a DRIFT GUARD. It is computed on rows where the reference and
the incumbent already agreed, so it measures whether the adapter is walking
away from what the base already got right. Near-flat is the expected and
desired result; it is not evidence of learning, and the train loss is the one
that should move.

RECIPE, and where each number came from

    r=16, alpha=32, dropout 0.05      sized to 402 rows / 589 items. The
                                      correction signal does not scale with
                                      epochs or rank: 68 of 589 items overturn
                                      the incumbent, 521 confirm it, and the
                                      base is already the qualified production
                                      judge, so the 521 teach it nothing.
    targets q/k/v/o + gate/up/down    #12, measured: the Gated DeltaNet
                                      projections won zero held-out examples
                                      at 58.2M extra params and ~15% slower.
    tower frozen                      #20's default; and under LoRA what
                                      matters is which modules are TARGETED.
    lr 1e-4 cosine, warmup 0.1        on record since 2026-09-15.
    3 epochs                          ~25 optimizer steps/epoch at effective
                                      batch 16, ~75 total.
    max_seq_len 10240                 #21, measured over all 402 rows: max is
                                      8,496 and 8192 truncates two of them --
                                      dropping the verdict, not padding.
"""

import json
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, TrainerCallback

from model import disable_kv_cache, fsdp_config, load_vision_model
from tasks.judge import dataset as judge_dataset
from tasks.judge.collator import JudgeCollator, JudgeDataset
from tasks.judge.masking import assert_masking_is_sane

# See judge_step_probe.py for why this list is duplicated rather than imported,
# and for the debt that creates.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class EpochLossRecorder(TrainerCallback):
    """
    Keep per-epoch train and eval loss, because the run is asked to report them.

    Trainer's log history holds this already, but it is a list of dicts that
    interleaves step logs, eval logs and the final summary, and "the per-epoch
    losses" should not be something the next reader has to reconstruct.
    """

    def __init__(self):
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.epoch is None:
            return
        if "loss" in logs:
            self.epochs.append({"epoch": round(state.epoch, 3),
                                "step": state.global_step, "train_loss": logs["loss"]})
        if "eval_loss" in logs:
            self.epochs.append({"epoch": round(state.epoch, 3),
                                "step": state.global_step, "eval_loss": logs["eval_loss"]})


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
    output_dir = os.environ.get(
        "OUTPUT_DIR", f"{templates_dir}/output/qwen3.8-27b-judge-rc0-lora"
    )
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    is_main = rank == 0

    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "3"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "10240"))
    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2"))
    n_eval = int(os.environ.get("N_EVAL_ROWS", str(judge_dataset.DEFAULT_N_EVAL)))

    rows = judge_dataset.load_rows(dataset_dir)
    train_rows, eval_rows = judge_dataset.train_eval_split(rows, dataset_dir, n_eval)

    if is_main:
        n_gpus = int(os.environ.get("WORLD_SIZE", "8"))
        print(f"Model       : {model_path}")
        print(f"Dataset     : {dataset_dir}")
        print(f"Output      : {output_dir}")
        print(f"Rows        : {len(train_rows)} train + {len(eval_rows)} "
              f"agreed-only drift guard = {len(rows)}")
        print(f"LoRA        : r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"Targets     : {LORA_TARGET_MODULES}")
        print(f"Batch       : {per_device_bs}/GPU x {grad_accum} accum x {n_gpus} GPUs "
              f"= effective {per_device_bs * grad_accum * n_gpus}")
        print(f"Epochs      : {num_epochs}")
        print(f"LR          : {learning_rate} cosine, warmup 0.1")
        print(f"Max seq len : {max_seq_len}")
        print(f"Eval rows   : {[r['id'][:8] for r in eval_rows]}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        # transformers v5: a value in [0, 1) is a ratio of total steps, so this
        # is exactly the old warmup_ratio=0.1. See train_lora.py.
        warmup_steps=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        # NO gradient_checkpointing here. Activation checkpointing comes from
        # fsdp_config(), which sets activation_checkpointing=True, and
        # transformers refuses both at once (trainer.py:845, job 1001):
        #     "The activation_checkpointing in FSDP config and the
        #      gradient_checkpointing in training arg can't be set to True
        #      simultaneously."
        #
        # The FSDP path is also the proven one on this stack, not merely the
        # recommended one: train_lora.py has always run LoRA under exactly this
        # fsdp_config, and that is what produced the 87.4% SQL result. So the
        # frozen-base adapters do receive gradients through it.
        #
        # judge_step_probe.py still sets gradient_checkpointing with
        # use_reentrant=False, and correctly: it runs on ONE GPU with no FSDP,
        # so there is no activation_checkpointing to conflict with, and under
        # the reentrant implementation the adapters would get no gradient at
        # all. The two paths differ because their parallelism differs.
        ddp_timeout=7200,
        fsdp=True,
        fsdp_config=fsdp_config(state_dict_type="FULL_STATE_DICT"),
        # ~75 optimizer steps in total, so every one of them is worth a line.
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        # The features are dicts of tensors the collator understands, not
        # columns Trainer should be pruning against the model signature.
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        report_to="none",
    )

    # Validate the config before anything expensive. TrainingArguments(bf16=True)
    # cannot be constructed on the login node, so this is the only place a
    # config error can be caught cheaply -- and it is built BEFORE the 52GB
    # load deliberately, so PREFLIGHT=1 costs seconds on one GPU. Job 1001 died
    # 90s in on a conflict between gradient_checkpointing and fsdp_config's
    # activation_checkpointing, which TrainingArguments accepts and Trainer
    # rejects one layer deeper.
    if os.environ.get("PREFLIGHT"):
        from cluster.preflight import validate

        sys.exit(validate(training_args, note="judge LoRA"))

    processor = judge_dataset.load_processor(model_path)

    # #22's gate, before the model is even built. Rank 0 checks every training
    # row; the others check a sample, which is enough to catch an environment
    # difference between nodes without paying the full pass eight times.
    t0 = time.perf_counter()
    assert_masking_is_sane(
        processor, train_rows, dataset_dir, sample=None if is_main else 20
    )
    if is_main:
        print(f"\nMasking gate: {len(train_rows)}/{len(train_rows)} rows pass "
              f"({time.perf_counter() - t0:.0f}s)")

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
    disable_kv_cache(model)

    adapted_vision = [
        n for n, _ in model.named_modules()
        if ("lora_A" in n or "lora_B" in n) and ".visual." in n
    ]
    assert not adapted_vision, (
        f"{len(adapted_vision)} LoRA adapters landed on the vision tower; it must "
        "stay frozen. See judge_step_probe.assert_vision_is_unadapted()."
    )

    # use_reentrant=False is mandatory, not stylistic: under the reentrant
    # implementation the checkpointed segment's inputs do not require grad --
    # the base weights are frozen and only the adapters train -- so NO gradient
    # reaches the LoRA parameters and the loss sits perfectly flat. It looks
    # exactly like a data problem. Measured in #23.
    model.enable_input_require_grads()

    if is_main:
        model.print_trainable_parameters()
        print("vision tower: 0 adapters (asserted)")


    recorder = EpochLossRecorder()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=JudgeDataset(processor, train_rows, dataset_dir),
        eval_dataset=JudgeDataset(processor, eval_rows, dataset_dir),
        data_collator=JudgeCollator(
            pad_token_id=processor.tokenizer.pad_token_id, max_seq_len=max_seq_len
        ),
        callbacks=[recorder],
    )

    if is_main:
        print("\nStarting judge LoRA training...")
    t_train = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t_train

    trainer.save_model(output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)

        # Provenance the cluster's own records do not carry (chat3d #95).
        record = {
            "base": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "dataset_dir": dataset_dir,
            "dataset_repo": "danielkreuzhofer/chat3d-judge-sft-rc0@54df8b59",
            "dataset_sha256": (
                "15ffadc0ee9317a1acdc821d3685b7e91dd52fdb5ca49e1470ca6298e28874e7"
            ),
            "rows_train": len(train_rows),
            "rows_eval_drift_guard": len(eval_rows),
            "eval_row_ids": [r["id"] for r in eval_rows],
            "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
                     "targets": LORA_TARGET_MODULES, "tower": "frozen"},
            "optim": {"lr": learning_rate, "schedule": "cosine", "warmup_ratio": 0.1,
                      "epochs": num_epochs, "max_grad_norm": 1.0,
                      "weight_decay": 0.01},
            "batch": {"per_device": per_device_bs, "grad_accum": grad_accum,
                      "world_size": int(os.environ.get("WORLD_SIZE", "8"))},
            "max_seq_len": max_seq_len,
            "train_seconds": round(train_seconds, 1),
            "losses": recorder.epochs,
            "log_history": trainer.state.log_history,
            "peak_gib_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_gib_reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        }
        with open(os.path.join(output_dir, "recipe.json"), "w") as handle:
            json.dump(record, handle, indent=1)

        print(f"\nTraining complete in {train_seconds / 60:.1f} min. "
              f"Adapter -> {output_dir}")
        print(f"Peak GPU mem (rank 0): {record['peak_gib_allocated']:.2f} GiB "
              f"allocated, {record['peak_gib_reserved']:.2f} GiB reserved")
        for row in recorder.epochs:
            print("  " + json.dumps(row))

    # Every rank waits for rank 0's write. Without this the others exit, their
    # CUDA contexts tear down under rank 0 mid-write, and the adapter is left
    # as an unrenamed .tmp file -- measured in job 811 on the SQL path.
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
