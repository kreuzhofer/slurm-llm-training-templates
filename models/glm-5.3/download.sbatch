#!/bin/bash
# =============================================================================
# download.sbatch -- fetch GLM-5.3 weights to shared storage as a one-node,
# CPU-only job.
#
#   sbatch download.sbatch [fp8|nvfp4|both]          # default: fp8
#
# Chain the server on it so it starts only after a clean download:
#   DL=$(sbatch --parsable download.sbatch nvfp4)
#   sbatch --dependency=afterok:$DL --nodes=2 serve.sbatch nvfp4
#
# No GPUs are requested, so this does not hold a GPU node exclusively.
# Re-running is cheap: completed files are skipped.
# =============================================================================

#SBATCH --job-name=glm53-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/glm53_download_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/glm53_download_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_activate_venv

echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Variant: ${1:-${VARIANT:-fp8}}"
# One progress bar per file is unreadable in a job log.
HF_HUB_DISABLE_PROGRESS_BARS=1 MAX_WORKERS="${SLURM_CPUS_PER_TASK:-16}" \
    bash "$TEMPLATES_DIR/repo/models/glm-5.3/download.sh" "$@"
