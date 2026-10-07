# slurm-llm-training-templates

Working templates for large language models on a Slurm cluster: fine-tuning,
evaluation and serving, on one node or several. Each template lives in its own
directory under `models/` and comes with a README that walks through it end to
end.

Every number in this repo is measured on real hardware, and every non-obvious
setting carries a comment saying why it is there.

## Templates

| template | what it covers | nodes |
|---|---|---|
| [Qwen3.8-27B](models/qwen3.8-27b/README.md) | LoRA and full fine-tuning for text-to-SQL and a multimodal judge, merge, evaluation, serving | 2 for training, 1 for evaluation and serving |
| [GLM-5.3](models/glm-5.3/README.md) | serving with vLLM, FP8 or NVFP4 weights, access through an SSH tunnel | 1 or 2 |

Measured results for the Qwen3.8-27B template are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Requirements

The templates are verified on 2 × 8 NVIDIA B300 SXM6 (sm_103, 268.6 GiB HBM
each), Slurm 25.11.3 via
[Soperator](https://github.com/nebius/nebius-solutions-library/tree/main/soperator)
on Nebius, driver 580.159.04 and CUDA 13.0.

- A filesystem shared by the login node and all compute nodes. The workspace,
  with the Python environment, model weights, datasets and job logs, is
  `/mnt/data/slurm-llm-templates`.
- Internet access to download models and datasets from Hugging Face.

Every cluster-specific setting is in [`cluster/env.sh`](cluster/env.sh). Read
it before running the templates on another cluster.

## Quick start

```bash
git clone https://github.com/kreuzhofer/slurm-llm-training-templates.git
cd slurm-llm-training-templates
bash cluster/setup.sh          # Python environment and repo copy on shared storage, then a 1-GPU check
source /mnt/data/slurm-llm-templates/activate.sh
```

Then follow the README of the template you want to run.

Slurm jobs run the copy of the repo under `/mnt/data/slurm-llm-templates/repo/`.
After changing files in your checkout, run `bash cluster/setup.sh` again to
update that copy.

## Repository layout

```
cluster/   settings shared by every job, environment setup, config preflight
models/    one directory per template: Slurm jobs, model code, README
tasks/     datasets, prompts and metrics used by the training templates
docs/      measured results
```

To add a template or a task, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
