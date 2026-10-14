#!/bin/bash
# =============================================================================
# setup.sh -- the LTX-2 inference stack in its own venv. Run once on the login
# node, after cluster/setup.sh.
#
#   bash models/ltx-2.5/setup.sh
#
# Creates $TEMPLATES_DIR/venv-ltx (see requirements.txt for why it is separate
# from the shared venv), clones Lightricks/LTX-2 at a pinned tag into
# $TEMPLATES_DIR/src/LTX-2, installs ltx-core and ltx-pipelines from it, builds
# ltx-kernels, and checks the result on one GPU. Re-running is safe.
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
VENV_DIR="$TEMPLATES_DIR/venv-ltx"
LTX_SRC="$TEMPLATES_DIR/src/LTX-2"
# v1.4.2 is the release LTX-2.5 runs on; main past it only changes license text.
LTX_REF="${LTX_REF:-v1.4.2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== LTX-2.5 setup ==="
echo "venv  : $VENV_DIR"
echo "source: $LTX_SRC @ $LTX_REF"

# --- Step 1: LTX-2 source at a pinned tag -------------------------------------
# Editable installs below point into this tree, so it must be on the shared
# filesystem: the workers import from it.
mkdir -p "$TEMPLATES_DIR/src"
if [ ! -d "$LTX_SRC/.git" ]; then
    git clone -q https://github.com/Lightricks/LTX-2.git "$LTX_SRC"
fi
git -C "$LTX_SRC" fetch -q --tags origin
git -C "$LTX_SRC" checkout -q "$LTX_REF"

# --- Step 2: venv and pinned stack --------------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip setuptools wheel
echo ""
echo "Installing pinned stack (several minutes) ..."
"$VENV_DIR/bin/pip" install -r "$HERE/requirements.txt"
"$VENV_DIR/bin/pip" install -e "$LTX_SRC/packages/ltx-core" -e "$LTX_SRC/packages/ltx-pipelines"

# --- Step 3: ltx-kernels -------------------------------------------------------
# The multi-GPU pipelines (distilled_mgpu, dfr_mgpu) need its all2all kernel for
# sequence parallelism; there is no fallback.
#
# TORCH_CUDA_ARCH_LIST=10.3 builds native sm_103 code for B300. Unset, torch
# picks archs from the GPUs it sees, and the login node has none.
#
# CUDA_HOME is left unset on purpose: setup.py then finds the pip nvcc 13.0.
# Exporting the system /usr/local/cuda (12.9) makes torch refuse the build.
echo ""
echo "Building ltx-kernels for sm_103 (several minutes) ..."
env -u CUDA_HOME -u CUDA_PATH \
    TORCH_CUDA_ARCH_LIST="10.3" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    LTX_KERNELS_CACHE_DIR="$TEMPLATES_DIR/src/ltx-kernels-cache" \
    PATH="$VENV_DIR/bin:$PATH" \
    "$VENV_DIR/bin/pip" install --no-build-isolation -e "$LTX_SRC/packages/ltx-kernels"

# --- Step 4: check on one GPU -------------------------------------------------
echo ""
echo "=== Checking the stack on one B300 (1-GPU job) ==="
srun --partition="${PARTITION:-main}" --nodes=1 --ntasks=1 --gpus-per-node=1 \
     --time=00:10:00 \
     "$VENV_DIR/bin/python" - <<'PYEOF'
import torch, torch.nn.functional as F
print("torch          :", torch.__version__)
cap = torch.cuda.get_device_capability(0)
print("device         :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
q = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
from torch.nn.attention import sdpa_kernel, SDPBackend
for backend in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION):
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(q, q, q)
    print(f"sdpa {backend.name:16s}:", bool(torch.isfinite(out).all()))
import natten
print("natten         :", natten.__version__)
import ltx_kernels.all_to_all  # noqa: F401  sequence-parallel all2all
print("ltx_kernels    : all2all ok")
import ltx_pipelines.distilled, ltx_pipelines.distilled_mgpu  # noqa: F401
import transformers
print("transformers   :", transformers.__version__)
print("ltx_pipelines  : import ok")
PYEOF

echo ""
echo "=== LTX-2.5 setup complete ==="
echo "Next: sbatch $TEMPLATES_DIR/repo/models/ltx-2.5/download.sbatch"
