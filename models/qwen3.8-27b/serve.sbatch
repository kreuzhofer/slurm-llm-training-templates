#!/bin/bash
# =============================================================================
# serve.sbatch -- vLLM OpenAI-compatible server for Qwen3.8-27B.
#
# HOW TO USE:
#   # serve the merged fine-tuned model (1 GPU, port 8000):
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/serve.sbatch
#
#   # serve the base model alongside it for A/B comparison:
#   sbatch /mnt/data/slurm-llm-templates/repo/models/qwen3.8-27b/serve.sbatch \
#       /mnt/data/slurm-llm-templates/models/Qwen3.8-27B 8001
#
# TP=1 is the default and is sufficient: ~50GB of bf16 weights on a 268.6 GiB B300.
# The KV cache is unusually cheap for a 27B model because only 16 of the 64
# layers cache at all -- the other 48 are linear attention, which keeps a
# small fixed-size recurrent state per sequence instead.
#
# If you do want tensor parallelism, the head counts constrain the valid sizes:
# 24 query / 4 KV / 16 linear-key / 48 linear-value, all read from config.json
# and confirmed against the downloaded checkpoint.
#
# TP=16 fails on the **24 query heads**, not the linear-key count -- 16 divides
# 16 exactly, so the earlier note here had the wrong reason. TP=3 and TP=6 also
# fail, on key_dim=2048, after passing every head-count check. TP in {1,2,4,8}
# is the safe set.
#
# Still unverified on this cluster: only TP=1 has been run. There is also a
# reported silent post-load hang at TP=4 for Qwen3.5/3.6 models, though on
# different hardware and an older vLLM. Test before relying on multi-GPU.
#
# QUERYING:
#   squeue --me                # NODELIST tells you which worker it landed on
#   bash models/qwen3.8-27b/query.sh
# =============================================================================

#SBATCH --job-name=qwen38-serve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=main
# Both partitions have TIMELIMIT=infinite, so WITHOUT this a forgotten serve job
# holds a B300 until someone notices and scancels it -- and the symptom is a
# training job queueing behind a server nobody is using. A limit turns that into
# "my server died", which is loud, recoverable, and blames the right thing.
# Override for a longer session:  sbatch --time=12:00:00 serve.sbatch ...
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/serve_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/serve_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_activate_venv
cluster_triton_cache

MODEL_PATH=${1:-$TEMPLATES_DIR/output/qwen3.8-27b-sql}
PORT=${2:-8000}
TP_SIZE=${3:-1}
MAX_MODEL_LEN=${4:-8192}   # native is 262144; 8192 is ample for SQL and starts faster

# Triton/torch.compile caches on shared NFS race between concurrent jobs.

echo "=========================================="
echo "Model     : $MODEL_PATH"
echo "Node      : $(hostname)"
echo "Port      : $PORT"
echo "TP size   : $TP_SIZE"
echo "Max len   : $MAX_MODEL_LEN"
echo ""
echo "Health    : curl http://$(hostname):$PORT/v1/models"
echo "=========================================="

# Flag notes:
#   --reasoning-parser qwen3
#       Effectively mandatory. The chat template opens every assistant turn
#       with <think>, so without a reasoning parser the whole reasoning block
#       lands in message.content and any exact-match scoring reads garbage.
#   --default-chat-template-kwargs '{"enable_thinking": false}'
#       Matches how the model was fine-tuned. Without it the template injects
#       a "reasoning effort is set to xhigh" system message and the model
#       reasons at length before answering -- slow, and it can overrun
#       max_tokens before emitting any SQL.
#   --language-model-only
#       VERIFIED against vllm 0.28.0 on this cluster: accepted, and it shows up
#       as 'language_model_only': True in the server's parsed non-default args.
#       It does NOT "skip building the vision tower to free KV cache" as this
#       comment used to claim -- it disables multimodal *inputs*. The flag that
#       skips vision profiling is --skip-mm-profiling.
#       On a MERGED checkpoint it is a no-op either way: merging writes
#       model_type qwen3_5_text with no vision_config, so there is no vision
#       tower to disable.
#
#       On the BASE checkpoint it is NOT harmless, which an earlier version of
#       this comment got wrong. The base config.json declares its own
#       "language_model_only": false next to
#       architectures: ["Qwen3_5ForConditionalGeneration"], so passing the flag
#       overrides the checkpoint's stated intent, and it is the one thing
#       preventing this server from accepting images. It is kept here only
#       because this script serves the SQL/text path, where images are not
#       wanted. ANY VISUAL SERVING MUST DROP IT.
#
#       Measured with the flag omitted on the base model (issue #17, job 842):
#       the tower is built (`FLASH_ATTN for vit attention`,
#       `MMEncoderAttention`, and a 16,384-token encoder cache) and it costs
#       almost nothing -- model load 51.1 GiB vs 50.13 GiB text-only (+0.97
#       GiB), and GPU KV cache 1,885,605 vs 1,899,098 tokens (-0.71%).
#       --skip-mm-profiling is confirmed present in vllm 0.28.0.
# --- Decide the vision flag FROM THE CHECKPOINT ------------------------------
# The comment above said "ANY VISUAL SERVING MUST DROP IT" and then the script
# dropped it for nobody: --language-model-only was passed unconditionally, and
# MODEL_PATH is just a positional argument. Serving the multimodal judge through
# this script therefore produced a judge that could not see the renders it
# exists to judge, with nothing in the log saying so.
#
# A comment cannot enforce that. The checkpoint already carries the answer --
# vision_config is present iff the tower is there, which is the same signal
# merge_judge_lora.py asserts on -- so read it instead of trusting the caller.
#
# The detection answers "can this checkpoint see", not "does this task want
# images", and it deliberately fails toward omitting the flag. The two mistakes
# are not symmetric: omitting it on a text workload costs 0.97 GiB and 0.71% of
# the KV budget (measured, #17) and changes no output, while passing it on a
# vision workload makes the model blind and says nothing. So the cheap mistake
# is the default.
#
# SERVE_MULTIMODAL=0 forces the flag back on -- worth it to reclaim that memory
# when serving a vision-capable checkpoint for a text-only task, e.g. scoring
# the base model on SQL. SERVE_MULTIMODAL=1 forces it off. Either is also the
# escape hatch when config.json cannot be read, e.g. a Hub id rather than a
# local directory.
LM_ONLY=()
if [ -n "${SERVE_MULTIMODAL:-}" ]; then
    mm="$SERVE_MULTIMODAL"
    echo "Multimodal  : $mm (forced via SERVE_MULTIMODAL)"
else
    set +e
    "$TEMPLATES_DIR/venv/bin/python" - "$MODEL_PATH" <<'PYEOF'
import json, os, sys
try:
    cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))
except Exception as exc:
    print(f"cannot read config.json: {exc}", file=sys.stderr)
    sys.exit(2)
sys.exit(0 if "vision_config" in cfg else 1)
PYEOF
    rc=$?
    set -e
    case "$rc" in
        0) mm=1 ;;
        1) mm=0 ;;
        *) echo "" >&2
           echo "Refusing to guess whether $MODEL_PATH is multimodal." >&2
           echo "Serving a vision model with --language-model-only silently" >&2
           echo "disables image input. Set SERVE_MULTIMODAL=1 or 0." >&2
           exit 1 ;;
    esac
    echo "Multimodal  : $mm (from config.json: vision_config $([ "$mm" = 1 ] && echo present || echo absent))"
fi
if [ "$mm" = 0 ]; then
    LM_ONLY=(--language-model-only)
else
    echo "              omitting --language-model-only so image input works"
fi

vllm serve "$MODEL_PATH" \
    --served-model-name "$(basename "$MODEL_PATH")" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    "${LM_ONLY[@]}" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
