#!/bin/bash
# =============================================================================
# download.sh -- fetch the MiniMax H3 files the diffusers pipeline needs.
#
#   bash download.sh
#
# Normally run as a queued job through download.sbatch. It also runs directly
# on a login node with internet egress:
#   source /mnt/data/slurm-llm-templates/activate.sh
#   bash /mnt/data/slurm-llm-templates/repo/models/minimax-h3/download.sh
#
# MiniMaxAI/MiniMax-H3 holds the same model twice: the original checkpoints
# (FL2VA/, Ref2VA/, 288 GB, for SGLang and vLLM-Omni) and the diffusers layout
# at the top level. diffusers needs only the second:
#
#   transformer/       61.7 GiB   text-to-video and first/last-frame (t2va, fl2va)
#   transformer_ref/   61.7 GiB   reference-to-video (ref2va)
#   text_encoder/      62.1 GiB   Qwen3-VL-32B, shared by all three
#   vae/, audio_vae/   10.3 GiB
#
# 210 GB in total, into $TEMPLATES_DIR/models/MiniMax-H3 with the Hub's folder
# layout. Re-running resumes: completed files are skipped.
#
# The weights are under the MiniMax H3 Community License, which does not cover
# the EU, the UK, the US or South Korea; see the README before downloading.
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
REPO=MiniMaxAI/MiniMax-H3
DEST="$TEMPLATES_DIR/models/MiniMax-H3"
mkdir -p "$DEST" "$TEMPLATES_DIR/logs"

# LICENSE and docs/ (the prompt-writing guides) are kept next to the weights.
INCLUDE=(
    model_index.json modular_model_index.json LICENSE "docs/*"
    "transformer/*" "transformer_ref/*" "text_encoder/*"
    "tokenizer/*" "processor/*" "vae/*" "audio_vae/*"
    "scheduler/*" "audio_scheduler/*"
)

# Refuse up front if what is still missing will not fit, rather than filling
# the shared filesystem half-way.
NEED_GB=$("$TEMPLATES_DIR/venv/bin/python" - "$REPO" "$DEST" "${INCLUDE[@]}" <<'PYEOF'
import fnmatch, os, sys
from huggingface_hub import HfApi
repo, dest, patterns = sys.argv[1], sys.argv[2], sys.argv[3:]
need = 0
for s in HfApi().model_info(repo, files_metadata=True).siblings:
    if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns) \
            and not os.path.exists(os.path.join(dest, s.rfilename)):
        need += s.size or 0
print(-(-need // 10**9))   # ceil, so 0 stays 0
PYEOF
)
AVAIL_GB=$(df -B1G --output=avail "$TEMPLATES_DIR" | tail -1 | tr -dc '0-9')
echo "Still to download: ${NEED_GB} GB   free: ${AVAIL_GB} GB"
if [ "$NEED_GB" -ge "$AVAIL_GB" ]; then
    echo "Not enough space on $(df --output=target "$TEMPLATES_DIR" | tail -1)." >&2
    exit 1
fi

# --include takes one pattern per flag. `--include a b c` makes b and c file
# names, and the CLI then ignores --include, so a is never fetched.
INCLUDE_ARGS=()
for p in "${INCLUDE[@]}"; do
    INCLUDE_ARGS+=(--include "$p")
done
echo "=== $REPO -> $DEST"
hf download "$REPO" "${INCLUDE_ARGS[@]}" --local-dir "$DEST" \
    --max-workers "${MAX_WORKERS:-16}"

echo "Done. Next:  sbatch $TEMPLATES_DIR/repo/models/minimax-h3/generate.sbatch prompts/en-lighthouse.txt"
