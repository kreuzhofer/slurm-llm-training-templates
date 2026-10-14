#!/bin/bash
# =============================================================================
# download.sh -- fetch the LTX-2.5 files the distilled and DFR pipelines need.
#
#   bash download.sh
#
# Normally run as a queued job through download.sbatch. It also runs directly
# on a login node with internet egress:
#   source /mnt/data/slurm-llm-templates/activate.sh
#   bash /mnt/data/slurm-llm-templates/repo/models/ltx-2.5/download.sh
#
# Both Hub repos are gated: accept the LTX-2.x Community License on each model
# page with the account behind your HF token first, or the download fails with
# "Access denied. This repository requires approval."
#
#   Lightricks/LTX-2.5                                  71 GB of the 201 GB repo
#   Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler   0.3 GB, DFR only
#
# Everything lands in $TEMPLATES_DIR/models/LTX-2.5 with the Hub's folder
# layout (diffusion_models/, text_encoders/, vae/, ...). Re-running resumes.
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
DEST="$TEMPLATES_DIR/models/LTX-2.5"
mkdir -p "$DEST" "$TEMPLATES_DIR/logs"

# The full repo also carries the dev transformer, int8/NVFP4 ComfyUI variants
# and an int8 text encoder. The distilled and DFR pipelines need only these.
FILES=(
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
    vae/ltx-2.5-video-vae-bf16.safetensors
    vae/ltx-2.5-audio-vae-bf16.safetensors
    model_patches/ltx-2.5-duration-head-bf16.safetensors
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
    latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors
)
DETAILING_REPO=Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler
DETAILING_FILE=ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors

# Refuse up front if what is still missing will not fit, rather than filling
# the shared filesystem half-way. This also fails fast on a gated repo whose
# license was not accepted.
NEED_GB=$("$TEMPLATES_DIR/venv/bin/python" - "$DEST" "${FILES[@]}" <<PYEOF
import os, sys
from huggingface_hub import HfApi
dest, files = sys.argv[1], set(sys.argv[2:])
need = 0
for repo, wanted, sub in (("Lightricks/LTX-2.5", files, ""),
                          ("$DETAILING_REPO", {"$DETAILING_FILE"}, "loras")):
    info = HfApi().model_info(repo, files_metadata=True)
    for s in info.siblings:
        if s.rfilename in wanted and not os.path.exists(os.path.join(dest, sub, s.rfilename)):
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

echo "=== Lightricks/LTX-2.5 -> $DEST"
hf download Lightricks/LTX-2.5 "${FILES[@]}" --local-dir "$DEST" \
    --max-workers "${MAX_WORKERS:-16}"
echo "=== $DETAILING_REPO -> $DEST/loras"
hf download "$DETAILING_REPO" "$DETAILING_FILE" --local-dir "$DEST/loras"

echo "Done. Next:  sbatch $TEMPLATES_DIR/repo/models/ltx-2.5/generate.sbatch"
