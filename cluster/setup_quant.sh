#!/bin/bash
# =============================================================================
# cluster/setup_quant.sh -- a SEPARATE venv for quantization, beside the training one.
#
#   bash cluster/setup_quant.sh
#
# WHY A SECOND VENV
#
# llm-compressor pins compressed-tensors 0.19.0. The training venv carries
# compressed-tensors 0.17.0, and that package is `Required-by: vllm` -- vllm 0.28.0
# was validated against it, and it is the read side of every quantized checkpoint
# we serve. Installing llm-compressor into the training venv would also bump
# accelerate and datasets. None of that is worth risking on the stack a release
# candidate trains on, days before it trains.
#
# So this venv carries the SAME torch, torchvision and transformers pins as the
# training venv, from the same index, and differs only in the quantizer. The
# training venv is never touched. Nothing here is installed on the login node's
# system Python either.
#
# torchvision is not optional: AutoProcessor on this checkpoint constructs a
# Qwen3VLVideoProcessor, which imports torchvision at load time, so the processor
# cannot even be opened without it. Found the hard way (job 1092).
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
TRAIN_VENV="$TEMPLATES_DIR/venv"
QUANT_VENV="$TEMPLATES_DIR/venv-quant"

[ -x "$TRAIN_VENV/bin/python" ] || {
    echo "training venv not found at $TRAIN_VENV; run cluster/setup.sh first" >&2
    exit 1
}

# Pin to whatever the training venv has, read from it rather than restated here,
# so the two cannot drift apart when requirements.txt moves.
pin() { "$TRAIN_VENV/bin/pip" show "$1" 2>/dev/null | awk '/^Version/{print $2}'; }
TORCH=$(pin torch); TORCHVISION=$(pin torchvision); TRANSFORMERS=$(pin transformers)
echo "pinning to the training venv: torch=$TORCH torchvision=$TORCHVISION transformers=$TRANSFORMERS"

if [ ! -x "$QUANT_VENV/bin/python" ]; then
    python3 -m venv "$QUANT_VENV"
fi
"$QUANT_VENV/bin/pip" install --quiet --upgrade pip
"$QUANT_VENV/bin/pip" install --quiet \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "torch==$TORCH" "torchvision==$TORCHVISION" "transformers==$TRANSFORMERS" \
    llmcompressor

# Prove the pins held and the quantizer imports. llm-compressor warns about no
# accelerator on the login node; that is expected and harmless here.
"$QUANT_VENV/bin/python" - <<PY 2>&1 | grep -v "No accelerator"
import torch, torchvision, transformers, llmcompressor, compressed_tensors
assert torch.__version__ == "$TORCH", torch.__version__
assert transformers.__version__ == "$TRANSFORMERS", transformers.__version__
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__)
print("llmcompressor", llmcompressor.__version__,
      "| compressed_tensors", compressed_tensors.__version__)
PY
echo "quant venv ready at $QUANT_VENV"
echo "  training venv compressed-tensors: $(pin compressed-tensors)  (unchanged)"
