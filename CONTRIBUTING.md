# Contributing

## Three axes, on purpose

```
cluster/          THE CLUSTER -- shared by every model and task
  env.sh          every env setting and the defect that justifies it
  preflight.py    validate a training config in seconds, not 16 GPU-minutes
  setup.sh        venv on shared NFS, repo sync, GPU smoke test

tasks/            THE TASK -- what is being measured
  sql/            text-to-SQL; exact match after normalisation
  judge/          multimodal CAD judge; 8 renders in, JSON verdict out

models/           THE ARCHITECTURE -- how a model loads, wraps, adapts and serves
  qwen3.8-27b/    hybrid attention + vision tower + MTP head
  glm-5.3/        inference only: vLLM on 1 or 2 nodes, FP8 or NVFP4

docs/             results, measurements, agent conventions
```

A task defines **what is being measured**; change it and that task's numbers
move, which is what keeps runs comparable. A model directory holds what is
**architecture-shaped**: how weights load, how FSDP wraps them, which modules
LoRA targets, how the model is served. The cluster layer holds what is true of
the **machine**.

Tasks never import from each other. Shared "convenience" code between tasks
ends up as one task wearing a neutral name.

## Working on the cluster

Slurm jobs execute `$TEMPLATES_DIR/repo/`, not your checkout. Re-run
`bash cluster/setup.sh` after editing anything: it rsyncs `cluster/`, `models/`
and `tasks/` there. Submitting without re-syncing runs the old code.

## Adding a template

Copy `models/<nearest>/` and change what is architecture-shaped.

Every template:

1. **`README.md`** -- a walkthrough for the person running it: download,
   start, check, use, stop. Measured numbers belong here; the story of how a
   setting was found belongs in a comment next to the setting.
2. **`download.sh` / `download.sbatch`** -- fetch weights to
   `$TEMPLATES_DIR/models/`, never into the repo.
3. **`*.sbatch`** -- source `cluster/env.sh` and call `cluster_setup`. Do not
   re-declare NCCL, Triton or bytecode settings; if one needs to change, it
   changes for every model at once, which is the entire reason that file
   exists.

Training templates also need:

4. **`model.py`** -- the load path, the FSDP wrap classes, which modules LoRA
   targets. Do not guess the target module names; enumerate them from the
   checkpoint. A suffix-matched list from an older model generation can
   silently miss modules or catch the wrong ones, e.g. a vision tower that uses
   `qkv`/`proj` rather than `q_proj`/`k_proj`.
5. **`train_*.py`** -- build `TrainingArguments` **before** loading the
   checkpoint and hand them to `cluster.preflight.validate` under `PREFLIGHT`.
   A config error then costs seconds instead of a full allocation.
6. **Verify what the merge writes.** Compare the merged checkpoint's tensor
   **name set** against the base, not its count and not the part you were
   worried about.

## Adding a task

Create `tasks/<name>/`. It owns its data loading, its prompt, its label masking
and its metric, and it imports from no other task.

If the task is multimodal, assume the masking is wrong until proven otherwise.
Token arithmetic over text undercounts a prompt by ~4,600 positions per row once
image placeholders expand, and nothing about that crashes -- the run completes,
the loss curve looks plausible, and the adapter learned on the wrong positions.
`tasks/judge/masking.py` shows the shape of a fix: define the boundary as the
inference-time prompt's own length, then verify every row and prove the check
fails on a deliberately wrong offset.

## The rule this repo keeps relearning

**A check that names the part you were worried about is blind to the part nobody
was worried about.** The vision-tower verification passed at 333 of 333 while
the same merge dropped the MTP head. A set difference has no such blind spot.
Prefer checks that compare against a reference wholesale over checks that assert
a specific expectation.
