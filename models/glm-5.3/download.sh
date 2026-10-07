#!/bin/bash
# =============================================================================
# download.sh -- fetch GLM-5.3 weights to shared storage.
#
#   bash download.sh [fp8|nvfp4|both]          # default: fp8
#
# Normally run as a queued job through download.sbatch, which passes its
# argument through. It also runs directly on a login node with internet egress:
#   source /mnt/data/slurm-llm-templates/activate.sh
#   bash /mnt/data/slurm-llm-templates/repo/models/glm-5.3/download.sh nvfp4
#
#   variant  Hub repo               local dir                       size
#   fp8      zai-org/GLM-5.3        $TEMPLATES_DIR/models/GLM-5.3        755.6 GB
#   nvfp4    nvidia/GLM-5.3-NVFP4   $TEMPLATES_DIR/models/GLM-5.3-NVFP4  464.1 GB
#
# NVFP4 runs only on Blackwell GPUs. Re-running resumes: completed files are
# skipped. CONFIG_ONLY=1 fetches config + tokenizer only (seconds), which is
# enough for a LOAD_FORMAT=dummy launch test of serve.sbatch.
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
VARIANT="${1:-${VARIANT:-fp8}}"
case "$VARIANT" in
    fp8)   REPOS=(zai-org/GLM-5.3) ;;
    nvfp4) REPOS=(nvidia/GLM-5.3-NVFP4) ;;
    both)  REPOS=(zai-org/GLM-5.3 nvidia/GLM-5.3-NVFP4) ;;
    *) echo "usage: $0 [fp8|nvfp4|both]   (got '$VARIANT')" >&2; exit 2 ;;
esac
mkdir -p "$TEMPLATES_DIR/models" "$TEMPLATES_DIR/logs"

dest() { echo "$TEMPLATES_DIR/models/$(basename "$1")"; }

if [ "${CONFIG_ONLY:-0}" = 1 ]; then
    for repo in "${REPOS[@]}"; do
        hf download "$repo" --exclude '*.safetensors' --local-dir "$(dest "$repo")"
        echo "Config and tokenizer for $repo in $(dest "$repo") (no weights)."
    done
    exit 0
fi

# Refuse up front if everything still missing will not fit, rather than
# filling the shared filesystem and failing half-way.
NEED_GB=$("$TEMPLATES_DIR/venv/bin/python" - "$TEMPLATES_DIR/models" "${REPOS[@]}" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi
root, repos = sys.argv[1], sys.argv[2:]
need = 0
for repo in repos:
    info = HfApi().model_info(repo, files_metadata=True)
    local = os.path.join(root, os.path.basename(repo))
    need += sum(s.size or 0 for s in info.siblings
                if not os.path.exists(os.path.join(local, s.rfilename)))
print(-(-need // 10**9))   # ceil, so 0 stays 0
PYEOF
)
AVAIL_GB=$(df -B1G --output=avail "$TEMPLATES_DIR" | tail -1 | tr -dc '0-9')
echo "Variant: $VARIANT   still to download: ${NEED_GB} GB   free: ${AVAIL_GB} GB"
if [ "$NEED_GB" -ge "$AVAIL_GB" ]; then
    echo "Not enough space on $(df --output=target "$TEMPLATES_DIR" | tail -1)." >&2
    exit 1
fi

for repo in "${REPOS[@]}"; do
    echo "=== $repo -> $(dest "$repo")"
    hf download "$repo" --local-dir "$(dest "$repo")" --max-workers "${MAX_WORKERS:-16}"
done
echo "Done. Next:  sbatch $TEMPLATES_DIR/repo/models/glm-5.3/serve.sbatch $([ "$VARIANT" = both ] && echo fp8 || echo "$VARIANT")"
