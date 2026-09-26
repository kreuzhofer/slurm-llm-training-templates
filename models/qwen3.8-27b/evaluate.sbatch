#!/bin/bash
# =============================================================================
# evaluate.sbatch -- score a base model against one or more fine-tunes on the
# held-out SQL test split.
#
# One GPU is plenty: 27B in bf16 is ~52GB against 268.6 GiB of HBM, and the models
# are loaded one at a time.
#
# Every argument is passed straight through to evaluate.py, so any flag it
# accepts works here. This used to take three positional arguments and forward
# only three flags, which meant --prefix could not be set -- so the documented
# full-fine-tune comparison silently overwrote the LoRA one's results and
# labelled them as LoRA.
#
# HOW TO USE:
#   # base vs merged LoRA, 500 examples (the default):
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/evaluate.sbatch
#
#   # three-way: base vs merged LoRA vs full fine-tune, one chart, base
#   # generated once:
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/evaluate.sbatch \
#       --tuned-model /mnt/data/slurm-llm-templates/output/qwen3.8-27b-sql \
#       --tuned-model /mnt/data/slurm-llm-templates/output/qwen3.8-27b-sql-full
#
#   # cheap smoke:
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/evaluate.sbatch --num-examples 100
#
# Results land in /mnt/data/slurm-llm-templates/results/, under a filename prefix
# derived from the tuned model names unless --prefix says otherwise.
# =============================================================================

#SBATCH --job-name=qwen38-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=main
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/evaluate_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/evaluate_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_activate_venv

# With no arguments, score the base model against the merged LoRA checkpoint.
# evaluate.py owns every other default, including --num-examples.
if [ "$#" -eq 0 ]; then
    set -- --base-model "$TEMPLATES_DIR/models/Qwen3.8-27B" \
           --tuned-model "$TEMPLATES_DIR/output/qwen3.8-27b-sql"
fi

echo "=========================================="
echo "evaluate.py $*"
echo "=========================================="

python "$TEMPLATES_DIR/repo/tasks/sql/evaluate.py" "$@"
