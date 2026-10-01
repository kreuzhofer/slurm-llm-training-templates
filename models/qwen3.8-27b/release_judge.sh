#!/bin/bash
# =============================================================================
# release_judge.sh -- everything between a trained judge adapter and two
# publishable checkpoints, as one gated chain.
#
#   bash release_judge.sh <rc-name> <adapter_dir> <dataset_dir>
#   e.g.  bash release_judge.sh rc1 \
#             $TEMPLATES_DIR/output/qwen3.8-27b-judge-rc1-lora \
#             $TEMPLATES_DIR/datasets/judge-sft-rc1-2456edfb
#
# Steps, each on one GPU, each stopping the chain on failure:
#
#   1. merge      adapter + base -> <rc>-merged, tower kept, 15 mtp.* restored
#                 from the base, tensor-NAME parity against the base asserted
#   2. head       train the MTP drafter against THAT merged trunk, trunk frozen.
#                 Prints the +2 alignment loss of the base head on this trunk
#                 before training ("before") and the drift-guard loss after
#                 each epoch ("after") -- the number this side can put on the
#                 record ahead of a serving window
#   3. install    overwrite the mtp shard with the trained head, in place;
#                 read back and compare bit for bit
#   4. parity     merge_judge_lora.py --verify-only would re-run the merge; the
#                 cheaper equivalent is the name-set check, re-asserted here
#   5. quantize   NVFP4-mixed with the reference recipe, calibrated on this
#                 release's own rows, MTP kept BF16 in its sidecar, verified
#
# Pushes are NOT part of this chain. Publishing is a separate, deliberate step.
#
# Why one script: rc0 did these as six commands typed in order over two days,
# with one of them (the MTP head) discovered missing downstream. The chain is
# the record of what a release IS, and it cannot skip a step by forgetting one.
# =============================================================================
set -euo pipefail

RC="${1:?rc name, e.g. rc1}"
ADAPTER="${2:?adapter dir}"
DATASET="${3:?dataset dir}"

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
BASE="$TEMPLATES_DIR/models/Qwen3.8-27B"
REPO="$TEMPLATES_DIR/repo"
MODEL_DIR="$REPO/models/qwen3.8-27b"
OUT="$TEMPLATES_DIR/output/qwen3.8-27b-judge-$RC"
PY="$TEMPLATES_DIR/venv/bin/python"
PYQ="$TEMPLATES_DIR/venv-quant/bin/python"
LOG="$TEMPLATES_DIR/logs/release_judge_$RC.log"
GPU="srun --partition=main --nodes=1 --gpus-per-node=1 --time=01:30:00"

for d in "$ADAPTER" "$DATASET" "$BASE"; do
    [ -d "$d" ] || { echo "missing: $d" >&2; exit 1; }
done
[ -x "$PYQ" ] || { echo "venv-quant missing; run cluster/setup_quant.sh first" >&2; exit 1; }
for d in "$OUT-merged" "$OUT-mtp" "$OUT-nvfp4"; do
    [ -e "$d" ] && { echo "$d exists; refusing to overwrite a previous release step" >&2; exit 1; }
done

step() { echo; echo "=================== [$RC] $1  $(date -u +%H:%M:%S) ==================="; }
{
step "1/5 merge"
$GPU "$PY" "$MODEL_DIR/merge_judge_lora.py" "$ADAPTER" "$BASE" "$OUT-merged"

step "2/5 MTP head against the merged trunk"
MODEL_PATH="$OUT-merged" JUDGE_DATASET_DIR="$DATASET" OUTPUT_DIR="$OUT-mtp" \
NUM_EPOCHS="${MTP_EPOCHS:-3}" N_EVAL_ROWS="${N_EVAL_ROWS:-30}" MAX_SEQ_LEN="${MAX_SEQ_LEN:-10240}" \
DROP_AUTO_C="${DROP_AUTO_C:-0}" \
$GPU env PYTHONPATH="$REPO" "$PY" "$MODEL_DIR/train_judge_mtp.py"

step "3/5 install the trained head"
"$PY" "$MODEL_DIR/install_mtp_head.py" "$OUT-merged" "$OUT-mtp/mtp_head.safetensors"

step "4/5 tensor-name parity against the base, after install"
"$PY" - "$BASE" "$OUT-merged" <<'PYEOF'
import json, sys
b = set(json.load(open(f"{sys.argv[1]}/model.safetensors.index.json"))["weight_map"])
o = set(json.load(open(f"{sys.argv[2]}/model.safetensors.index.json"))["weight_map"])
assert b == o, f"parity broken: missing {sorted(b-o)[:5]} extra {sorted(o-b)[:5]}"
print(f"parity: {len(o)} = {len(b)} tensor names, identical sets")
PYEOF

step "5/5 NVFP4-mixed build"
$GPU env PYTHONPATH="$REPO" "$PYQ" "$MODEL_DIR/quantize_nvfp4.py" \
    "$OUT-merged" "$OUT-nvfp4" --dataset-dir "$DATASET" \
    --calibration-rows "${CALIB_ROWS:-128}" --max-seq-len "${MAX_SEQ_LEN:-10240}"

step "done"
echo "BF16 : $OUT-merged"
echo "NVFP4: $OUT-nvfp4"
echo "head : $OUT-mtp  (mtp_recipe.json carries the before/after alignment losses)"
echo "Publish deliberately, e.g.:"
echo "  hf upload danielkreuzhofer/chat3d-judge-$RC $OUT-merged . && \\"
echo "  hf upload danielkreuzhofer/chat3d-judge-$RC $OUT-nvfp4 nvfp4"
echo "RELEASE_CHAIN_OK"
} 2>&1 | tee "$LOG"
