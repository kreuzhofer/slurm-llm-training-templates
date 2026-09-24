#!/bin/bash
# =============================================================================
# train_judge_lora.sbatch -- rc0 judge LoRA, one node x 8 B300 (#24).
#
# HOW TO USE:
#   sbatch /mnt/data/qwen38-demo/repo/models/qwen3.8-27b/train_judge_lora.sbatch
#
# MONITORING:
#   squeue --me
#   tail -f /mnt/data/qwen38-demo/logs/train_judge_lora_<JOBID>.out
#
# ONE NODE, not two. The recipe on record is 8 GPUs, the set is 402 rows, and
# ~75 optimizer steps do not need a second node's interconnect in the picture.
# =============================================================================

#SBATCH --job-name=qwen38-judge-rc0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # torchrun forks the 8 GPU processes
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/qwen38-demo/logs/train_judge_lora_%j.out
#SBATCH --error=/mnt/data/qwen38-demo/logs/train_judge_lora_%j.err

set -euo pipefail

export DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
source "$DEMO_DIR/venv/bin/activate"

export MODEL_PATH="$DEMO_DIR/models/Qwen3.8-27B"
# Repo head 54df8b59, sha256 15ffadc0…, 402 rows / 589 items. Named for the
# head so it cannot be confused with the superseded 152-row upload beside it.
export JUDGE_DATASET_DIR="${JUDGE_DATASET_DIR:-$DEMO_DIR/datasets/judge-sft-rc0-54df8b59}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEMO_DIR/output/qwen3.8-27b-judge-rc0-lora}"

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
# 10240, not 8192: measured max is 8,496 over the 402 rows, and 8192 truncates
# two of them -- dropping the verdict, not padding (#21).
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-10240}"
# Agreed-only drift guard. Not a holdout: rc0's evaluation is the qualification
# screen on the held-out 125, on chat3d's side.
export N_EVAL_ROWS="${N_EVAL_ROWS:-10}"

# --- Carried from the SQL runs; not optional on this cluster ------------------
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_TIMEOUT=7200

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29501

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_JOB_NODELIST"
echo "GPUs       : $SLURM_GPUS_ON_NODE"
echo "Model      : $MODEL_PATH"
echo "Dataset    : $JUDGE_DATASET_DIR"
echo "Output     : $OUTPUT_DIR"
echo "=========================================="

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$DEMO_DIR/repo/models/qwen3.8-27b/train_judge_lora.py"
