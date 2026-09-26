#!/bin/bash
# =============================================================================
# train_lora.sbatch -- LoRA fine-tune Qwen3.8-27B on the SQL task, 2 nodes x 8.
#
# HOW TO USE:
#   sbatch $TEMPLATES_DIR/repo/models/qwen3.8-27b/train_lora.sbatch
#
# CHECK THE CONFIG FIRST -- one GPU, seconds, no checkpoint load. Must be
# torchrun, not bare python: fsdp=True cannot be validated outside a distributed
# context. This reproduces job 1001's failure in seconds instead of 90s into an
# 8-GPU allocation.
#   PREFLIGHT=1 srun --nodes=1 --gpus-per-node=1 --partition=main --time=10 \
#       "$TEMPLATES_DIR/venv/bin/torchrun" --nnodes=1 --nproc_per_node=1 \
#       --rdzv_backend=c10d --rdzv_endpoint=localhost:29777 \
#       "$TEMPLATES_DIR/repo/models/qwen3.8-27b/train_lora.py"
#
# MONITORING:
#   squeue --me
#   tail -f $TEMPLATES_DIR/logs/train_lora_<JOBID>.out
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
# =============================================================================

#SBATCH --job-name=qwen38-sql-lora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one task per node; torchrun forks the 8 GPU procs
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/train_lora_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/train_lora_%j.err

set -euo pipefail

# Slurm copies the batch script to its own spool directory, so the repo root
# cannot be derived from this file's location. It comes from the environment
# with one default, and cluster/env.sh carries every setting and the reason for
# it -- venv, NCCL, Triton cache, bytecode, and the lessons that are not
# environment variables at all. Read that file before changing anything here.
TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup

# --- What to train -----------------------------------------------------------
export MODEL_PATH="$TEMPLATES_DIR/models/Qwen3.8-27B"
export DATASET_PATH="$TEMPLATES_DIR/datasets/sql-create-context"
# Overridable so A/B arms write to separate directories without editing this
# file (see LORA_TARGET_GDN and #12).
export OUTPUT_DIR="${OUTPUT_DIR:-$TEMPLATES_DIR/output/qwen3.8-27b-sql-lora}"

# 0 = attention + MLP only. MEASURED in #12: adding the 48 Gated DeltaNet
# projections costs 58.2M trainable params and ~15% throughput and changes no
# predictions (A-only=0 at both 25 and 500 steps). Set to 1 to re-enable.
export LORA_TARGET_GDN="${LORA_TARGET_GDN:-0}"
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05

# 8 per GPU x 16 GPUs x 1 accum = effective batch 128, matching the Qwen3-era
# runs so the accuracy numbers stay comparable. Measured: 584 optimizer steps
# over the 74,648-example train split.
#
# There is headroom: B300 has 268.6 GiB HBM against roughly 3GB/rank of sharded
# frozen base weights, so activations dominate. Raise PER_DEVICE_BATCH_SIZE
# before reaching for gradient accumulation.
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_EPOCHS=1
export LEARNING_RATE=2e-4            # LoRA tolerates ~10x the full-FT rate
export MAX_SEQ_LEN=1024

# --- Short-run knobs ---------------------------------------------------------
# Defaults reproduce a full single-epoch run exactly. Override per submission:
#   MAX_STEPS=25 SAVE_STEPS=10 MAX_EVAL_EXAMPLES=200 sbatch --export=ALL ...
export MAX_STEPS="${MAX_STEPS:--1}"                   # -1 = run NUM_EPOCHS epochs
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-$SAVE_STEPS}"
export MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:--1}"   # -1 = all 3,929

# Sequence packing is unavailable for this architecture -- the Gated DeltaNet
# recurrent state cannot be reset mid-sequence -- so expect lower tokens/s per
# GPU than an equivalent dense Qwen3 run.

cluster_rendezvous 29500
cluster_banner
echo "Model      : $MODEL_PATH"
echo "Output     : $OUTPUT_DIR"

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$TEMPLATES_DIR/repo/models/qwen3.8-27b/train_lora.py"
