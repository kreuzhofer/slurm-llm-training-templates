#!/bin/bash
# =============================================================================
# judge_step_probe.sbatch -- one GPU, one repeated batch of judge rows (#23).
#
# HOW TO USE:
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/judge_step_probe.sbatch
#
# MONITORING:
#   squeue --me
#   tail -f /mnt/data/slurm-llm-templates/logs/judge_step_probe_<JOBID>.out
#
# ONE GPU, NO TORCHRUN, NO FSDP. The whole 27.36B model in bf16 is ~51 GiB
# against 268.6 GiB of HBM, so sharding would add a variable without buying
# anything -- and the step time this prints is meant to be read without
# wondering how much of it was collective communication.
# =============================================================================

#SBATCH --job-name=qwen38-judge-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=main
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/judge_step_probe_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/judge_step_probe_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup

export MODEL_PATH="$TEMPLATES_DIR/models/Qwen3.8-27B"
# The 402-row export, repo head 54df8b59. Named for the head rather than
# "judge-sft-rc0" so it cannot be confused with the superseded 152-row upload
# still sitting beside it.
export JUDGE_DATASET_DIR="${JUDGE_DATASET_DIR:-$TEMPLATES_DIR/datasets/judge-sft-rc0-54df8b59}"

# The recipe on record for rc0, so the step time measured here is the step time
# the real run will have.
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
# 10240, not 8192: measured max total is 8,496 over the 402 rows, and 8192
# truncates two of them -- dropping the verdict, not padding, because the
# assistant turn is last (#21).
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-10240}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export PROBE_STEPS="${PROBE_STEPS:-10}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

cluster_banner
echo "Model      : $MODEL_PATH"
echo "Dataset    : $JUDGE_DATASET_DIR"

srun python "$TEMPLATES_DIR/repo/models/qwen3.8-27b/judge_step_probe.py"
