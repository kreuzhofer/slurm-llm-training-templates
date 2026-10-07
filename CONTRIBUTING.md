# Contributing to slurm-llm-training-templates

Contributions are welcome: new templates, fixes to existing ones, and
measurements from other clusters. Please follow the
[Contributing Guidelines](#contributing-guidelines) below.

### Setup

```bash
bash cluster/setup.sh
source /mnt/data/slurm-llm-templates/activate.sh
```

Slurm jobs run the copy of the repo under `/mnt/data/slurm-llm-templates/repo/`,
not your checkout. Run `bash cluster/setup.sh` again after every change before
you submit a job.

### Pull Requests

1. Create your branch from `main`.
2. Run every job you changed on the cluster, on each layout it supports.
3. Update the template's README if the steps or the numbers changed.
4. Put the measured results in the PR description: job IDs, what you ran, and
   what came out.

### Issues

We use GitHub issues to track bugs. Include the `sbatch` command, the job ID,
the relevant part of the job log, and the cluster you ran on.

### License

By contributing, you agree that your contributions will be licensed under the
[LICENSE](LICENSE) file in the root of this repository.

---

## Contributing Guidelines

### Principles of contribution

- Keep the three layers separate:
  - `cluster/` holds what is true of the machine: environment, NCCL settings,
    paths.
  - `models/<template>/` holds what depends on the model: how it loads, how it
    is sharded or served, its Slurm jobs and its README.
  - `tasks/<task>/` holds what is being measured: data loading, prompt, label
    masking, metric.
- Every Slurm job sources `cluster/env.sh` and calls `cluster_setup`. Do not
  redefine NCCL, Triton or Python settings in a job; change them in `env.sh`
  for every template at once.
- Tasks do not import from each other.
- Weights, datasets, outputs and logs live under `$TEMPLATES_DIR` on the shared
  filesystem, never in the repo.
- Explain every non-obvious setting in a comment next to it, with the symptom
  that made it necessary.
- A template README is a walkthrough for the person running it. Keep the
  history of how a setting was found in the code comment, not in the README.

### Proof of Value

It is the contributor's responsibility to show that a change works. Numbers in
this repo are measured, not estimated.

#### Training templates

- Validate the config with `cluster/preflight.py` before using a full
  allocation. Build `TrainingArguments` before loading the checkpoint and pass
  them to `cluster.preflight.validate` under `PREFLIGHT`.
- Report the metric on held-out data, next to the base model on the same rows.
- After a merge, compare the merged checkpoint's tensor names against the base
  checkpoint as a set. Checking only the part you changed misses tensors that
  were dropped elsewhere.
- For multimodal tasks, check the label masking on every row, and show that
  the check fails on a deliberately wrong offset.

#### Inference templates

- Start the server on every layout the template supports, for example one node
  and two nodes, and with every weight variant.
- Check answers at temperature 0, tool calls, streaming, a long prompt, and a
  burst of concurrent requests followed by a single request.
- Report startup time and KV cache size from the server log.

### Best practices

- Start a new template by copying the nearest one in `models/`.
- Enumerate LoRA target modules from the checkpoint instead of reusing a list
  from another model.
- Give long-running jobs, especially servers, a `--time` limit.
- Prefer checks that compare against a reference as a whole over checks that
  assert one expected detail.
