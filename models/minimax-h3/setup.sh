#!/bin/bash
# =============================================================================
# setup.sh -- the MiniMax H3 inference stack in its own venv. Run once on the
# login node, after cluster/setup.sh.
#
#   bash models/minimax-h3/setup.sh
#
# Creates $TEMPLATES_DIR/venv-h3 (see requirements.txt for why it is separate
# from the shared venv), installs the pinned stack, and checks it on one GPU.
# Nothing is compiled. Re-running is safe.
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
VENV_DIR="$TEMPLATES_DIR/venv-h3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== MiniMax H3 setup ==="
echo "venv  : $VENV_DIR"

# --- Step 1: venv and pinned stack --------------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip setuptools wheel
echo ""
echo "Installing pinned stack (several minutes) ..."
"$VENV_DIR/bin/pip" install -r "$HERE/requirements.txt"

# --- Step 2: check on one GPU -------------------------------------------------
echo ""
echo "=== Checking the stack on one B300 (1-GPU job) ==="
srun --partition="${PARTITION:-main}" --nodes=1 --ntasks=1 --gpus-per-node=1 \
     --mem=32G --time=00:10:00 \
     "$VENV_DIR/bin/python" - <<'PYEOF'
import torch, torch.nn.functional as F
print("torch          :", torch.__version__)
cap = torch.cuda.get_device_capability(0)
print("device         :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
# H3 attends over one packed sequence with no mask, 56 heads of 128.
q = torch.randn(1, 56, 4096, 128, device="cuda", dtype=torch.bfloat16)
from torch.nn.attention import sdpa_kernel, SDPBackend
for backend in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION):
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(q, q, q)
    print(f"sdpa {backend.name:16s}:", bool(torch.isfinite(out).all()))
import av, diffusers, transformers
from diffusers import ModularPipeline  # noqa: F401
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference  # noqa: F401
print("diffusers      :", diffusers.__version__)
print("transformers   :", transformers.__version__)
print("av             :", av.__version__)
PYEOF

echo ""
echo "=== MiniMax H3 setup complete ==="
echo "Next: sbatch $TEMPLATES_DIR/repo/models/minimax-h3/download.sbatch"
