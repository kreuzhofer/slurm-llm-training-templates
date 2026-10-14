#!/bin/bash
# =============================================================================
# download.sbatch -- fetch the LTX-2.5 weights (71 GB) to shared storage as a
# one-node, CPU-only job.
#
#   sbatch download.sbatch
#
# Chain a generation on it so it starts only after a clean download:
#   DL=$(sbatch --parsable download.sbatch)
#   sbatch --dependency=afterok:$DL generate.sbatch prompts/en-interview.txt
#
# No GPUs are requested, so this does not hold a GPU node exclusively.
# Re-running is cheap: completed files are skipped.
# =============================================================================

#SBATCH --job-name=ltx25-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/ltx25_download_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/ltx25_download_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
# The shared venv has the `hf` CLI; venv-ltx is not needed to download.
cluster_activate_venv

echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
# One progress bar per file is unreadable in a job log.
HF_HUB_DISABLE_PROGRESS_BARS=1 MAX_WORKERS="${SLURM_CPUS_PER_TASK:-16}" \
    bash "$TEMPLATES_DIR/repo/models/ltx-2.5/download.sh"
