# slurm-llm-training-templates

Working templates for fine-tuning large language and vision-language models on a
Slurm cluster — LoRA and full-parameter, multi-node FSDP2, merge, serve, evaluate.

Every number in this repo is measured on real hardware, and every non-obvious
line of configuration carries the defect that put it there. That is the point of
the repo: the training loops are the easy part, and what actually costs days is
the cluster, the masking, the checkpoint layout, and the silent failures that
complete successfully with the wrong answer.

**Verified on** 2 × 8 NVIDIA B300 SXM6 (sm_103, 268.6 GiB HBM each), Slurm
25.11.3 via [Soperator](https://github.com/nebius/nebius-solutions-library/tree/main/soperator)
on Nebius, driver 580.159.04 / CUDA 13.0, shared filesystem at `/mnt/data`.
Nothing here is portable by assumption — read [`cluster/env.sh`](cluster/env.sh)
before running it anywhere else.

## Three axes, on purpose

```
cluster/          THE CLUSTER -- shared by every model and task
  env.sh          every env setting and the defect that justifies it
  preflight.py    validate a training config in seconds, not 16 GPU-minutes
  setup.sh        venv on shared NFS, repo sync, GPU smoke test

tasks/            THE TASK -- what is being measured
  sql/            text-to-SQL; exact match after normalisation
  judge/          multimodal CAD judge; 8 renders in, JSON verdict out

models/           THE ARCHITECTURE -- how a model loads, wraps, and adapts
  qwen3.8-27b/    hybrid attention + vision tower + MTP head
  glm-5.3/        inference only: vLLM on 1 or 2 nodes, FP8 or NVFP4

docs/             results, measurements, agent conventions
```

A task defines **what is being measured**; change it and that task's numbers
move, which is what keeps runs comparable. A model directory holds what is
**architecture-shaped**: how weights load, how FSDP wraps them, which modules
LoRA targets. The cluster layer holds what is true of the **machine**.

Tasks never import from each other. That is deliberate — shared "convenience"
code between tasks is how a repo ends up with a directory called `common/` that
is really one task wearing a neutral name, which is exactly what this repo had
and removed.

## Start here

| | |
|---|---|
| **Qwen3.8-27B** | [`models/qwen3.8-27b/README.md`](models/qwen3.8-27b/README.md) — SQL LoRA vs full fine-tune (87.4% vs 55.4% base), and a multimodal judge |
| **GLM-5.3 inference** | [`models/glm-5.3/README.md`](models/glm-5.3/README.md) — serve GLM-5.3 (FP8 or NVFP4) on 1 or 2 nodes, reach it through an SSH tunnel |
| **Full results** | [`docs/RESULTS.md`](docs/RESULTS.md) — every measurement, the charts, and the defects that running it exposed |
| **Cluster facts and lessons** | [`cluster/env.sh`](cluster/env.sh) |

```bash
git clone https://github.com/kreuzhofer/slurm-llm-training-templates.git
cd slurm-llm-training-templates
bash cluster/setup.sh          # venv on shared NFS, ends with a GPU smoke test
source /mnt/data/slurm-llm-templates/activate.sh
```

Re-run `bash cluster/setup.sh` after editing anything: it rsyncs `cluster/`,
`models/` and `tasks/` to `$TEMPLATES_DIR/repo/`, which is what the Slurm jobs
actually execute. Editing your checkout and submitting without re-syncing runs
the old code.

## Adding a model

Copy `models/<nearest>/` and change what is architecture-shaped. In practice:

1. **`model.py`** — the load path, the FSDP wrap classes, which modules LoRA
   targets. Do not guess the target module names; enumerate them from the
   checkpoint. On Qwen3.8 a Qwen3-era attention-only list reaches a quarter of
   the token-mixing blocks, and its vision tower uses `qkv`/`proj` rather than
   `q_proj`/`k_proj`, so a suffix-matched list silently misses or catches the
   wrong thing.
2. **`train_*.py`** — build `TrainingArguments` **before** loading the
   checkpoint and hand them to `cluster.preflight.validate` under `PREFLIGHT`.
   A config error then costs seconds instead of a full allocation.
3. **`*.sbatch`** — source `cluster/env.sh` and call `cluster_setup`. Do not
   re-declare NCCL, Triton or bytecode settings; if one needs to change, it
   changes for every model at once, which is the entire reason that file exists.
4. **Verify what the merge writes.** Compare the merged checkpoint's tensor
   **name set** against the base, not its count and not the part you were
   worried about. A merge on this architecture silently dropped 15 MTP tensors
   and a vision-tower check could not see it.

## Adding a task

Create `tasks/<name>/`. It owns its data loading, its prompt, its label masking
and its metric, and it imports from no other task.

If the task is multimodal, assume the masking is wrong until proven otherwise.
Token arithmetic over text undercounts a prompt by ~4,600 positions per row once
image placeholders expand, and nothing about that crashes — the run completes,
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

## License

MIT — see [LICENSE](LICENSE).
