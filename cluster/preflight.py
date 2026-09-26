"""
cluster/preflight.py -- validate a training config before it costs you 16 GPUs.

WHY THIS EXISTS

`TrainingArguments(bf16=True)` cannot be constructed on the login node, because
there is no GPU there. So on this cluster a config error is invisible until a
job starts, and "a job starts" can mean a two-node allocation that spends five
minutes loading a 52GB checkpoint before it gets to the line that was wrong.

That has happened twice here in different ways:

  * `warmup_ratio` was removed in transformers v5 and silently folded into
    `warmup_steps`. Discovering it cost a 16-GPU submission.
  * Job 1001 died 90 seconds in: TrainingArguments(gradient_checkpointing=True)
    cannot be combined with fsdp_config["activation_checkpointing"], and
    transformers raises from `Trainer.__init__` ->
    `create_accelerator_and_postprocess`, not from TrainingArguments itself.

The second one matters for the design here. A preflight that only constructs
TrainingArguments would NOT have caught it -- the arguments are individually
valid and the conflict is detected one layer deeper, when the Trainer builds its
Accelerator. So this constructs a real Trainer, around a parameter-sized-for-
nothing module, and throws it away.

HOW TO USE IT

The training script builds its own real TrainingArguments and then hands them
here, so there is no second copy of the config to drift:

    if os.environ.get("PREFLIGHT"):
        from cluster.preflight import validate
        sys.exit(validate(training_args, note="judge LoRA"))

Then, before submitting the real job -- under torchrun, with ONE process:

    PREFLIGHT=1 srun --nodes=1 --gpus-per-node=1 --partition=main --time=10 \\
        "$TEMPLATES_DIR/venv/bin/torchrun" --nnodes=1 --nproc_per_node=1 \\
        --rdzv_backend=c10d --rdzv_endpoint=localhost:29777 \\
        "$TEMPLATES_DIR/repo/models/<model>/train_<task>.py"

torchrun, not bare python, and that is not incidental. `fsdp=True` raises
"Using fsdp only works in distributed training" outside a distributed context,
so a bare single process cannot validate an FSDP config at all -- which is
exactly the config worth validating. One process under torchrun sets
RANK/WORLD_SIZE, the FSDP plugin is constructed for real, and the
gradient_checkpointing conflict that killed job 1001 is reachable.

One GPU, no checkpoint load, seconds. It is not a substitute for a smoke run --
it validates configuration, not the data path or the loss -- but it is the
cheapest possible check and it catches the class of failure that wastes the most
GPU-minutes per occurrence.
"""

import sys


def validate(training_args, note="", verbose=True):
    """
    Build a Trainer around `training_args` and report. Returns a process exit code.

    Deliberately does not load the real model: the point is to reach the code
    paths that inspect the CONFIG -- accelerator construction, FSDP plugin
    setup, strategy validation -- which is where the expensive mistakes live,
    and none of them need real weights.
    """
    import torch
    from torch import nn

    if verbose:
        label = f" ({note})" if note else ""
        print(f"preflight{label}: constructing Trainer from the real config ...")

    # Small enough to be free, real enough that Trainer's checks engage: it
    # needs at least one trainable parameter to build an optimizer.
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(8, 8)

        def forward(self, input_ids=None, labels=None, **_):
            x = self.w(torch.zeros(1, 8, device=self.w.weight.device,
                                   dtype=self.w.weight.dtype))
            return {"loss": x.sum()}

    from transformers import Trainer

    # Trainer cross-checks args against what it was given: eval_strategy other
    # than "no" without an eval_dataset is a ValueError, and the same for a
    # load_best_model_at_end without eval. Stubs keep those checks meaningful --
    # the point is to validate the CONFIG, so a missing stub would report a
    # problem with the preflight rather than with the config under test.
    stub = [{"input_ids": [0], "labels": [0]}]

    try:
        Trainer(model=_Tiny(), args=training_args,
                train_dataset=stub, eval_dataset=stub)
    except Exception as exc:  # noqa: BLE001 -- reporting is the whole job
        print(f"\nPREFLIGHT FAILED: {type(exc).__name__}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("\nThe real job would have failed the same way, after its "
              "allocation and checkpoint load.", file=sys.stderr)
        return 1

    if verbose:
        fsdp = getattr(training_args, "fsdp", None)
        cfg = getattr(training_args, "fsdp_config", None) or {}
        print("  Trainer constructed OK")
        print(f"  bf16={training_args.bf16}  fsdp={fsdp}")
        print(f"  gradient_checkpointing={training_args.gradient_checkpointing}"
              f"  fsdp activation_checkpointing={cfg.get('activation_checkpointing')}")
        print(f"  warmup_steps={training_args.warmup_steps} "
              f"(a value in [0,1) is a RATIO of total steps in transformers v5)")
        print(f"  epochs={training_args.num_train_epochs} "
              f"per_device={training_args.per_device_train_batch_size} "
              f"accum={training_args.gradient_accumulation_steps}")
        print("\nPREFLIGHT OK -- config is constructible. This says nothing "
              "about your data path; still smoke-run before a real run.")
    return 0
