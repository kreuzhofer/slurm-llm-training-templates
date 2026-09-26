#!/bin/bash
# =============================================================================
# train_judge_lora.sbatch -- rc0 judge LoRA, one node x 8 B300 (#24).
#
# HOW TO USE:
#   sbatch $TEMPLATES_DIR/repo/models/qwen3.8-27b/train_judge_lora.sbatch
#
# CHECK THE CONFIG FIRST -- one GPU, seconds, no checkpoint load. Must be
# torchrun, not bare python: fsdp=True cannot be validated outside a distributed
# context. This reproduces job 1001's failure in seconds instead of 90s into an
# 8-GPU allocation.
#   PREFLIGHT=1 srun --nodes=1 --gpus-per-node=1 --partition=main --time=10 \
#       "$TEMPLATES_DIR/venv/bin/torchrun" --nnodes=1 --nproc_per_node=1 \
#       --rdzv_backend=c10d --rdzv_endpoint=localhost:29777 \
#       "$TEMPLATES_DIR/repo/models/qwen3.8-27b/train_judge_lora.py"
#
# MONITORING:
#   squeue --me
#   tail -f $TEMPLATES_DIR/logs/train_judge_lora_<JOBID>.out
#
# ONE NODE, not two. The recipe on record is 8 GPUs and 402 rows; ~75 optimizer
# steps do not need a second node's interconnect in the picture.
# =============================================================================

#SBATCH --job-name=qwen38-judge-rc0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # torchrun forks the 8 GPU processes
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/train_judge_lora_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/train_judge_lora_%j.err

set -euo pipefail

# See train_lora.sbatch for why the root comes from the environment rather than
# from this file's location, and cluster/env.sh for every setting and its reason.
TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup

export MODEL_PATH="$TEMPLATES_DIR/models/Qwen3.8-27B"
# Repo head 54df8b59, sha256 15ffadc0..., 402 rows / 589 items. Named for the
# head so it cannot be confused with the superseded 152-row upload beside it.
export JUDGE_DATASET_DIR="${JUDGE_DATASET_DIR:-$TEMPLATES_DIR/datasets/judge-sft-rc0-54df8b59}"
export OUTPUT_DIR="${OUTPUT_DIR:-$TEMPLATES_DIR/output/qwen3.8-27b-judge-rc0-lora}"

# --- The recipe on record ----------------------------------------------------
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
# 1 x 2 x 8 GPUs = effective 16, so ~25 optimizer steps/epoch over 392 train
# rows and ~75 in total.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
# 10240, not 8192: measured max total is 8,496 over the 402 rows, and 8192
# truncates two of them -- dropping the VERDICT, not padding, because the
# assistant turn is last (#21).
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-10240}"
# Agreed-only drift guard, not a holdout: rc0's evaluation is the qualification
# screen on the held-out 125, which lives on the chat3d side.
export N_EVAL_ROWS="${N_EVAL_ROWS:-10}"

cluster_rendezvous 29501
cluster_banner
echo "Model      : $MODEL_PATH"
echo "Dataset    : $JUDGE_DATASET_DIR"
echo "Output     : $OUTPUT_DIR"

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$TEMPLATES_DIR/repo/models/qwen3.8-27b/train_judge_lora.py"
