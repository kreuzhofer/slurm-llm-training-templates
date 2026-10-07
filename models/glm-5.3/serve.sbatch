#!/bin/bash
# =============================================================================
# serve.sbatch -- vLLM OpenAI-compatible server for GLM-5.3, 1 or N nodes.
#
# HOW TO USE:
#   sbatch [--nodes=2] serve.sbatch [fp8|nvfp4]          # default: fp8, 1 node
#
#   # one node, 8 x B300, TP=8 (FP8 is 704 GiB of weights; a node has 2.1 TiB):
#   sbatch /mnt/data/slurm-llm-templates/repo/models/glm-5.3/serve.sbatch
#
#   # two nodes, 16 x B300: TP=8 inside each node, PP=2 across them:
#   sbatch --nodes=2 /mnt/data/slurm-llm-templates/repo/models/glm-5.3/serve.sbatch fp8
#
#   # launch test WITHOUT the weights (random weights, gibberish output --
#   # proves kernels, memory, parallel layout, multi-node). Needs only
#   # CONFIG_ONLY=1 bash download.sh <variant>:
#   LOAD_FORMAT=dummy sbatch --export=ALL --nodes=2 --time=45 \
#       /mnt/data/slurm-llm-templates/repo/models/glm-5.3/serve.sbatch
#
#   variant  checkpoint                         served model name
#   fp8      $TEMPLATES_DIR/models/GLM-5.3        glm-5.3-fp8
#   nvfp4    $TEMPLATES_DIR/models/GLM-5.3-NVFP4  glm-5.3-nvfp4   (Blackwell only)
#
# QUERYING:
#   bash models/glm-5.3/query.sh "Say hello in one word."
#
# KNOBS (environment, all optional -- pass with sbatch --export=ALL):
#   MODEL_PATH     overrides the variant's checkpoint directory
#   PORT           default 8000, on the head node only
#   TP_SIZE        default = GPUs per node (8)
#   PP_SIZE        default = number of nodes. TP_SIZE*PP_SIZE must equal all GPUs.
#                  Keep TP inside a node: TP_SIZE=16 PP_SIZE=1 on two nodes hangs
#                  in vllm 0.28.0 during all-reduce setup (14 of 16 ranks never
#                  return), with nothing in the log after 20 minutes. Stage
#                  boundaries are set from config.json, see "Pipeline stage
#                  boundaries" below.
#   MAX_MODEL_LEN  default 131072; the model supports up to 1048576
#   MTP_TOKENS     default 5 (the vLLM recipe value); 0 disables MTP drafting.
#                  Forced to 0 when PP_SIZE > 1 (see below).
#   LOAD_FORMAT    default auto; "dummy" for the no-weights smoke test
#   SERVED_NAME    default glm-5.3-<variant> -- what clients put in "model"
#   EXTRA_ARGS     appended verbatim to `vllm serve`
#
# WHY vLLM's OWN MULTI-NODE LAUNCHER AND NOT RAY:
#   vllm 0.28.0 (the shared venv) runs multi-node TP/PP with its multiprocessing
#   backend: the same `vllm serve` on every node with --nnodes/--node-rank/
#   --master-addr, the head node serving HTTP and the others --headless. Ray is
#   not installed in the venv, and this needs nothing beyond one srun task per
#   node -- the same shape as train_full.sbatch.
#
# WHY TP INSIDE A NODE, PP ACROSS:
#   All 8 GPUs in a node share one NVSwitch domain (cluster/env.sh). TP does an
#   all-reduce in every layer, so it belongs on NVLink; PP sends one activation
#   tensor per micro-batch between stages, which is what crosses InfiniBand.
# =============================================================================

#SBATCH --job-name=glm53-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one vllm per node; it forks the 8 GPU workers
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
# Both partitions have TIMELIMIT=infinite (see the Qwen serve.sbatch). A
# forgotten 16-GPU server would hold the whole cluster. Override for longer:
#   sbatch --time=12:00:00 serve.sbatch
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/glm53_serve_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/glm53_serve_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup

VARIANT="${1:-${VARIANT:-fp8}}"
case "$VARIANT" in
    fp8)   DEFAULT_PATH="$TEMPLATES_DIR/models/GLM-5.3"
           VARIANT_ARGS="" ;;
    # vLLM recipe for NVFP4 on Blackwell: experts spread across GPUs (EP)
    # instead of every expert sliced 8 ways. The quantization method is read
    # from the checkpoint's config.json (modelopt mixed FP8/NVFP4).
    nvfp4) DEFAULT_PATH="$TEMPLATES_DIR/models/GLM-5.3-NVFP4"
           VARIANT_ARGS="--enable-expert-parallel" ;;
    *) echo "usage: sbatch [--nodes=N] serve.sbatch [fp8|nvfp4]   (got '$VARIANT')" >&2
       exit 2 ;;
esac
export VARIANT VARIANT_ARGS
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_PATH}"
export PORT="${PORT:-8000}"
export NNODES="$SLURM_NNODES"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
export TP_SIZE="${TP_SIZE:-$GPUS_PER_NODE}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MTP_TOKENS="${MTP_TOKENS:-5}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export SERVED_NAME="${SERVED_NAME:-glm-5.3-$VARIANT}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ $((TP_SIZE * PP_SIZE)) -ne $((NNODES * GPUS_PER_NODE)) ]; then
    echo "TP_SIZE*PP_SIZE = $((TP_SIZE * PP_SIZE)) but the job has" \
         "$((NNODES * GPUS_PER_NODE)) GPUs ($NNODES x $GPUS_PER_NODE)." >&2
    exit 1
fi

# --- Pipeline stage boundaries -----------------------------------------------
# GLM-5.3's sparse-attention indexer computes its top-k token selection only
# on "full" layers (0, 1, 2, 6, 10, ..., 74 -- config.json indexer_types); the
# three "shared" layers after each one reuse it through a per-process buffer.
# vLLM's default even split puts layer 39, a shared layer, first on stage 1.
# Its source, layer 38, is on stage 0, so stage 1 attends with whatever the
# buffer still holds from the previous step -- another batch's token indices.
# Measured on FP8, TP=8 x PP=2: the first request after 16 concurrent ones
# answered a question nobody asked in 5 of 16 trials (0 of 16 without the
# burst, 0 of 8 on one node). NVFP4 with PP returned runs of "!" for 17 of 56
# prompts. Starting every stage on a "full" layer removes the stale read.
if [ "$PP_SIZE" -gt 1 ]; then
    VLLM_PP_LAYER_PARTITION=$("$TEMPLATES_DIR/venv/bin/python" - "$MODEL_PATH" "$PP_SIZE" <<'PYEOF'
import json, os, sys
path = os.path.join(sys.argv[1], "config.json")
if not os.path.exists(path):
    sys.exit(f"{path} missing -- run download.sbatch for this variant first")
cfg = json.load(open(path))
pp = int(sys.argv[2])
types = cfg["indexer_types"]
n = len(types)
full = [i for i, t in enumerate(types) if t == "full"]
cuts = [0] + [min(full, key=lambda f: abs(f - k * n / pp)) for k in range(1, pp)] + [n]
if len(set(cuts)) != len(cuts):
    sys.exit(f"cannot split {n} layers into {pp} stages that each start on a full indexer layer")
print(",".join(str(b - a) for a, b in zip(cuts, cuts[1:])))
PYEOF
)
    export VLLM_PP_LAYER_PARTITION
    echo "Stages     : layers per stage $VLLM_PP_LAYER_PARTITION (each starts on a full indexer layer)"
fi

# FlashInfer autotuning at startup is a collective: every rank benchmarks the
# same kernel tactics and the timings are averaged across ALL ranks. With PP
# the stages hold different layers; NVFP4's three dense FP8 layers are only on
# stage 0, so stage 1 waits at the closing barrier until it times out after
# ~30 minutes ("Application timeout caused pair closure"). NVFP4 with PP
# therefore runs on FlashInfer's default kernel choices. FP8 with PP starts
# normally with tuning on, so it keeps it.
if [ "$PP_SIZE" -gt 1 ] && [ "$VARIANT" = nvfp4 ]; then
    echo "Note       : NVFP4 with PP_SIZE=$PP_SIZE, so FlashInfer autotuning is off."
    VARIANT_ARGS="$VARIANT_ARGS --no-enable-flashinfer-autotune"
fi

# vLLM 0.28.0 refuses MTP drafting together with pipeline parallelism: the
# GLM-5.3 model supports PP, its MTP draft model (DeepSeekMTPModel) does not,
# and startup fails with "Pipeline parallelism is not supported for this
# model". With PP > 1, MTP is switched off rather than failing the job.
if [ "$PP_SIZE" -gt 1 ] && [ "$MTP_TOKENS" -gt 0 ]; then
    echo "Note       : PP_SIZE=$PP_SIZE, so MTP speculative decoding is off" \
         "(not supported with pipeline parallelism)."
    MTP_TOKENS=0
fi

# --- Refuse a half-downloaded checkpoint -------------------------------------
# `hf download` writes each shard under .cache/ and renames it into place only
# when complete, so a missing name means a missing shard. Compare against the
# index wholesale -- "some *.safetensors exist" passes on a download that
# stopped at shard 3 of 141, and vLLM's error then is a KeyError deep in
# load_weights 10 minutes into the job.
if [ "$LOAD_FORMAT" != "dummy" ]; then
    "$TEMPLATES_DIR/venv/bin/python" - "$MODEL_PATH" <<'PYEOF'
import json, os, sys
root = sys.argv[1]
index = os.path.join(root, "model.safetensors.index.json")
if not os.path.exists(index):
    sys.exit(f"{index} missing -- run download.sbatch for this variant first")
shards = sorted(set(json.load(open(index))["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(root, s))]
if missing:
    sys.exit(f"{len(missing)} of {len(shards)} shards missing under {root}, "
             f"e.g. {missing[:3]} -- the download is incomplete")
print(f"Checkpoint : all {len(shards)} shards present")
PYEOF
fi

# --- Start-up cost -----------------------------------------------------------
# With DeepGEMM enabled, vLLM JIT-compiles a DeepGEMM kernel for every FP8
# linear shape at ~65 batch sizes before serving: ~3,300 kernels at ~2 s each,
# close to two hours on a cold cache. With it off, the FP8 linear layers run on
# vLLM's precompiled CUTLASS kernels instead. The sparse-attention indexer still
# uses DeepGEMM (a handful of kernels), and the MoE layers use FlashInfer either
# way. Opt back in with VLLM_USE_DEEP_GEMM=1 once its cache is warm.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# --- Head node address -------------------------------------------------------
# An IP, not a hostname: the address is handed to torch.distributed and to
# vLLM's message queues on every node, and both have to agree on it.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export HEAD_NODE
export MASTER_IP=$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1{print $1}')
# Port for vLLM's own torch.distributed group. Distinct from the 29500 the
# training jobs use, so a server and a training run never share a rendezvous.
export SERVE_DIST_PORT="${SERVE_DIST_PORT:-29511}"

cluster_banner
echo "Variant    : $VARIANT"
echo "Model      : $MODEL_PATH (load-format $LOAD_FORMAT)"
echo "Layout     : TP=$TP_SIZE x PP=$PP_SIZE over $NNODES node(s)"
echo "Head       : $HEAD_NODE ($MASTER_IP), torch port $SERVE_DIST_PORT"
echo "Max len    : $MAX_MODEL_LEN    MTP tokens: $MTP_TOKENS"
echo "Endpoint   : http://$HEAD_NODE:$PORT/v1   (model \"$SERVED_NAME\")"
echo "=========================================="

# --- One vllm per node --------------------------------------------------------
# The per-node script is single-quoted so that $SLURM_NODEID expands on each
# node rather than once here. --kill-on-bad-exit: if either node's vllm dies,
# take the other one down too instead of leaving a head waiting forever on a
# worker that is gone.
srun --kill-on-bad-exit=1 --ntasks-per-node=1 bash -c '
set -euo pipefail
# kv-cache dtype and both parsers are the recipe values from
# recipes.vllm.ai/zai-org/GLM-5.3 (B200 section). prefetch reads the shards
# sequentially into page cache first: vLLM enables that by itself only on
# filesystems it recognises as network storage (NFS, Lustre), and the
# memory-mapped fallback reads a 700 GiB checkpoint at ~400 MB/s.
# --block-size 64 is what one node picks by itself (the sparse-attention
# indexer wants 64). With PP each stage picks from its OWN first attention
# layer: stage 1 lands on the MLA backend, picks 32, and startup dies with
# "No common block size for 32". Pinning it keeps every stage at 64.
ARGS=(
    "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --pipeline-parallel-size "$PP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --load-format "$LOAD_FORMAT"
    --safetensors-load-strategy prefetch
    --block-size 64
    --kv-cache-dtype fp8_e4m3
    --reasoning-parser glm47
    --tool-call-parser glm47
    --enable-auto-tool-choice
)
if [ "$MTP_TOKENS" -gt 0 ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi
if [ "$NNODES" -gt 1 ]; then
    ARGS+=(--nnodes "$NNODES" --node-rank "$SLURM_NODEID"
           --master-addr "$MASTER_IP" --master-port "$SERVE_DIST_PORT")
fi
if [ "$SLURM_NODEID" -eq 0 ]; then
    # --master-addr and the Endpoint line both name HEAD_NODE, and query.sh
    # finds it as the job batch host. Fail loudly if Slurm ever puts node 0
    # elsewhere, rather than have the other node wait on a head that is not
    # there.
    if [ "$(hostname -s)" != "$HEAD_NODE" ]; then
        echo "node 0 is $(hostname -s), expected head $HEAD_NODE" >&2
        exit 1
    fi
    ARGS+=(--host 0.0.0.0 --port "$PORT")
else
    ARGS+=(--headless)
fi
# shellcheck disable=SC2206
ARGS+=($VARIANT_ARGS $EXTRA_ARGS)
echo "[node $SLURM_NODEID $(hostname)] vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
'
