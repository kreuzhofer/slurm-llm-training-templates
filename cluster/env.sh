#!/bin/bash
# =============================================================================
# cluster/env.sh -- every environment setting this cluster needs, and why.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../../cluster/env.sh"
#
# Sourced by every .sbatch in this repo. It exists because these settings were
# duplicated across six job scripts with one model in the tree; with three
# models that would be eighteen copies, and the failures below are exactly the
# kind that get lost when a fix lands in one copy and not the others.
#
# NOTHING HERE IS A PREFERENCE. Every line is a defect that cost a real job.
# If you are porting these templates to another cluster, read the reasons rather
# than copying the values -- most are Slurm-and-shared-filesystem facts, not
# Nebius facts, but the paths are not.
# =============================================================================

# --- Where everything lives --------------------------------------------------
# The workspace root on the SHARED filesystem: venv, model weights, datasets,
# job output, and the synced copy of this repo that jobs actually execute.
#
# DEMO_DIR is the old name, accepted so that an in-flight script or another
# agent session holding it keeps working. Prefer TEMPLATES_DIR in new code.
export TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
export DEMO_DIR="$TEMPLATES_DIR"

# /tmp IS NODE-LOCAL on this cluster. A worker rank cannot see what the
# launcher wrote there, and a multi-node job that stages anything into /tmp
# fails in a way that looks like a data problem. Shared artifacts go under
# $TEMPLATES_DIR. Write throughput there depends on file shape, measured:
# ~40 MB/s for many small files against ~214 MiB/s for large sequential shards.
cluster_assert_shared() {
    case "$1" in
        /tmp/*) echo "cluster/env.sh: $1 is node-local; use \$TEMPLATES_DIR" >&2
                return 1 ;;
    esac
}

# --- Python ------------------------------------------------------------------
# The venv lives on shared NFS so every worker node sees the same interpreter.
# Note `python` on a bare worker is /usr/bin/python with no torch and no peft;
# anything not launched through this file must use the venv interpreter by path.
cluster_activate_venv() {
    # shellcheck disable=SC1091
    source "$TEMPLATES_DIR/venv/bin/activate"
}

# Without this, every rank races to write .pyc files into the same
# __pycache__ on shared NFS while importing the task module. Same class of
# problem as TRITON_CACHE_DIR below.
export PYTHONDONTWRITEBYTECODE=1

# --- Per-job node-local scratch ----------------------------------------------
# Triton compiles kernels per rank. Pointed at shared NFS, 16 ranks write into
# one directory and corrupt each other's cache. $TMPDIR is node-local, which is
# exactly what is wanted here -- the opposite of the rule above, because this
# is a cache and not an artifact.
cluster_triton_cache() {
    export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-$$}"
    mkdir -p "$TRITON_CACHE_DIR"
}

# --- Interconnect ------------------------------------------------------------
# Measured on this cluster: 8 ConnectX-8 HCAs per node at 800 Gb/s, one PCIe
# hop (PIX) from each GPU, so GPUDirect RDMA has a dedicated NIC per rank. All
# 8 GPUs in a node are NV18 to each other -- one NVSwitch domain.
cluster_nccl_env() {
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"      # INFO to debug IB reachability
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
    # Loading a 52GB checkpoint across 16 ranks is hundreds of broadcasts, and
    # the 1800s default process-group timeout is not generous over shared NFS.
    # Pair with ddp_timeout in TrainingArguments, which is a SEPARATE timeout.
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
}

# --- Rendezvous --------------------------------------------------------------
# One port per concurrent job class, or two jobs rendezvous on the same port
# and one of them hangs until the timeout above.
cluster_rendezvous() {
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    MASTER_PORT="${1:-29500}"
    export MASTER_ADDR MASTER_PORT
}

# --- The standard preamble ---------------------------------------------------
# What almost every job wants, in one call.
cluster_setup() {
    cluster_activate_venv
    cluster_triton_cache
    cluster_nccl_env
}

cluster_banner() {
    echo "=========================================="
    echo "Job ID     : ${SLURM_JOB_ID:-n/a}"
    echo "Nodes      : ${SLURM_JOB_NODELIST:-n/a}"
    echo "GPUs/node  : ${SLURM_GPUS_ON_NODE:-n/a}"
    echo "Workspace  : $TEMPLATES_DIR"
    [ -n "${MASTER_ADDR:-}" ] && echo "Master     : $MASTER_ADDR:${MASTER_PORT}"
    echo "=========================================="
}

# =============================================================================
# LESSONS THAT ARE NOT ENVIRONMENT VARIABLES
#
# They belong with the settings above because they are the same kind of thing --
# things that cost a job on this cluster -- and because a reader porting these
# templates needs them in one place.
#
#  * TrainingArguments(bf16=True) CANNOT be constructed on the login node: no
#    GPU. Validating a config therefore costs a 1-GPU srun. cluster/preflight.py
#    does it, and it is cheaper than every alternative -- job 1001 died 90s in
#    on a config conflict that a login-node check could not have caught.
#
#  * Under FSDP, TrainingArguments(gradient_checkpointing=True) and
#    fsdp_config["activation_checkpointing"] cannot both be set; transformers
#    refuses. Use the fsdp_config one -- it is also the proven path here.
#    WITHOUT FSDP the reverse holds: use gradient_checkpointing with
#    use_reentrant=False, or a frozen-base LoRA gets no gradient at all and the
#    loss sits silently flat.
#
#  * End every distributed run with accelerator.wait_for_everyone(). Without
#    it, non-zero ranks exit while rank 0 is still writing, their CUDA contexts
#    tear down under it, and the checkpoint is left as an unrenamed .tmp file.
#
#  * Short runs find defects; they do not produce numbers. Three projections
#    extrapolated from smoke runs on this cluster were wrong by 2-5x. Peak
#    memory over 584 steps was 41% higher than over 25.
#
#  * Deleting anything under $TEMPLATES_DIR needs a human's explicit approval.
#    It holds the only local copy of ~50GB base checkpoints.
# =============================================================================
