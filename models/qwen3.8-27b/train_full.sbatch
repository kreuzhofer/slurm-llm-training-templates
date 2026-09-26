#!/bin/bash
# =============================================================================
# train_full.sbatch -- full-parameter fine-tune of Qwen3.8-27B on 2 nodes x 8 B300.
#
# All ~26.9B text parameters are trained, sharded with FSDP2. This is the
# variant the H100-era version of this demo could not run for a *smaller*
# model -- 268.6 GiB of HBM per GPU turns it into a routine job (~20-28GB/GPU for
# weights, gradients and optimizer state).
#
# HOW TO USE:
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/train_full.sbatch
#
# CHECK THE CONFIG FIRST -- one GPU, seconds, no checkpoint load. Must be
# torchrun, not bare python: fsdp=True cannot be validated outside a distributed
# context. This reproduces job 1001's failure in seconds instead of 90s into an
# 8-GPU allocation.
#   PREFLIGHT=1 srun --nodes=1 --gpus-per-node=1 --partition=main --time=10 \
#       "$TEMPLATES_DIR/venv/bin/torchrun" --nnodes=1 --nproc_per_node=1 \
#       --rdzv_backend=c10d --rdzv_endpoint=localhost:29777 \
#       "$TEMPLATES_DIR/repo/models/qwen3.8-27b/train_full.py"
#
#   # with mid-run checkpoints (costs a ~54GB gather+write each time):
#   SAVE_STRATEGY=steps SAVE_STEPS=200 sbatch --export=ALL \
#       /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/train_full.sbatch
#
# MONITORING:
#   squeue --me
#   tail -f /mnt/data/slurm-llm-templates/logs/train_full_<JOBID>.out
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
#
# The output checkpoint is directly usable -- no LoRA merge step:
#   sbatch models/qwen3.8-27b/evaluate.sbatch <base> <output>
#   sbatch models/qwen3.8-27b/serve.sbatch <output>
# =============================================================================

#SBATCH --job-name=qwen38-full
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one task per node; torchrun forks the 8 GPU procs
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/train_full_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/train_full_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup

# --- What to train -----------------------------------------------------------
export MODEL_PATH="$TEMPLATES_DIR/models/Qwen3.8-27B"
export OUTPUT_DIR="$TEMPLATES_DIR/output/qwen3.8-27b-sql-full"
export DATASET_PATH="$TEMPLATES_DIR/datasets/sql-create-context"

# 8 per GPU x 16 GPUs x 1 accum = effective batch 128, the same as
# train_lora.sbatch and as the Qwen3-era runs, so accuracy stays comparable.
# ~583 steps for 74.6k examples.
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_EPOCHS=1
export LEARNING_RATE=1e-5            # full FT wants ~1/20th of the LoRA rate
export MAX_SEQ_LEN=1024

# Checkpointing: "no" writes the model once at the end. See the long comment in
# train_full.py -- a FULL_STATE_DICT gather of a 27B model is ~54GB and doing it
# every few hundred steps is what blew the distributed timeout in the Qwen3-era
# pipeline. Override to trade I/O for resumability.

# --- Short-run knobs ---------------------------------------------------------
# Defaults reproduce a full single-epoch run exactly. Override per-submission:
#   MAX_STEPS=25 SAVE_STEPS=10 MAX_EVAL_EXAMPLES=200 \
#       sbatch --export=ALL scripts/train_full.sbatch
export MAX_STEPS="${MAX_STEPS:--1}"           # -1 = run NUM_EPOCHS epochs
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-$SAVE_STEPS}"
export MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:--1}"   # -1 = all 3,929

export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
# SAVE_STEPS is defined with the short-run knobs above; only SAVE_STRATEGY is
# full-FT-specific, because a 54GB FULL_STATE_DICT gather is not something you
# want happening every few hundred steps by default.

# FULL-FT-SPECIFIC: the final FULL_STATE_DICT gather of a 54GB model is a long
# collective and the NCCL watchdog kills it mid-save. cluster/env.sh sets the
# timeout; only disabling the monitor is specific to this path.
export TORCH_NCCL_ENABLE_MONITORING=0


# Sequence packing is unavailable for this architecture (the Gated DeltaNet
# recurrent state cannot be reset mid-sequence), so expect lower tokens/s per
# GPU than an equivalent dense Qwen3 run.

cluster_rendezvous 29500
cluster_banner
echo "Model      : $MODEL_PATH (full fine-tune)"
echo "Output     : $OUTPUT_DIR"
echo "Save       : $SAVE_STRATEGY"

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$TEMPLATES_DIR/repo/models/qwen3.8-27b/train_full.py"
