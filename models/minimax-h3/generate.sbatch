#!/bin/bash
# =============================================================================
# generate.sbatch -- one MiniMax H3 video with sound per job, on one GPU.
#
#   sbatch generate.sbatch prompts/en-lighthouse.txt     # prompt from a file
#   sbatch generate.sbatch "integrated_multimodal_description: [Shot 1] ..."
#
# Writes $TEMPLATES_DIR/output/minimax-h3/h3_<JOBID>.mp4 (H.264 + AAC stereo)
# and a copy of the prompt next to it, then exits.
#
# KNOBS (environment, all optional):
#   DURATION     seconds, 5 to 15, default 5. Rounded up to the next frame
#                count the VAE decodes (17n+5 frames at 24 fps): 5 -> 124
#                frames, 10 -> 243. 345 frames (14.4 s) is the longest; 15
#                gets that. Not SECONDS: bash counts that one up by itself.
#   ASPECT       16:9 (default without an image), 9:16, 1:1, 4:3, 21:9, ...
#                The short edge is 768: 16:9 is 1344x768, 9:16 is 768x1344.
#                With IMAGE the default is the image's own aspect.
#   SEED         default 42
#   STEPS        default 50. The checkpoints are guidance-distilled: one
#                forward pass per step, no negative prompt, no guidance scale.
#   IMAGE        first frame (fl2va)
#   LAST_IMAGE   last frame (fl2va); with IMAGE, the video goes from one to
#                the other
#   REFS         space-separated reference files (ref2va): images, videos,
#                audio. The prompt names them <Picture 1>, <Video 1>,
#                <Audio 1>, ... in the order given. Not with IMAGE/LAST_IMAGE.
#
# GPUs: one per video, bf16, everything resident (about 135 GB of weights on a
# 268 GiB B300), eight jobs side by side on a node. MiniMax's multi-GPU paths
# (SGLang, vLLM-Omni with Ulysses sequence parallelism) are servers that load
# once and take requests; this template renders one video per job instead.
# =============================================================================

#SBATCH --job-name=h3-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
# DefMemPerNode is UNLIMITED here: without --mem a job claims all 2.4 TB of the
# node and the other seven GPUs sit idle.
#SBATCH --mem=250G
#SBATCH --partition=main
# Both partitions have TIMELIMIT=infinite; a hung job would hold its GPU forever.
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/h3_generate_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/h3_generate_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup venv-h3
# Every component loads from the local copy. Offline, a component that would
# still resolve to the Hub fails at once instead of downloading 60 GB into the
# home directory's cache.
export HF_HUB_OFFLINE=1

if [ $# -ne 1 ]; then
    echo "usage: sbatch generate.sbatch <prompt-file | \"prompt text\">" >&2
    exit 2
fi
# sbatch copies the script to a spool directory, so a relative prompt path
# resolves against the submit directory, which is where the job starts.
# The file must be visible from the compute node: a prompt file under the login
# node's /tmp is not, and a single word with no file behind it is a path that
# did not resolve, not a prompt.
if [ -f "$1" ]; then
    PROMPT="$(cat "$1")"
elif [[ "$1" != *[[:space:]]* ]]; then
    echo "Prompt file '$1' not found on $(hostname). Put it on shared storage" \
         "(e.g. under \$TEMPLATES_DIR or your home directory)." >&2
    exit 2
else
    PROMPT="$1"
fi

DURATION="${DURATION:-5}"
SEED="${SEED:-42}"
STEPS="${STEPS:-50}"

OUT_DIR="$TEMPLATES_DIR/output/minimax-h3"
OUT="$OUT_DIR/h3_${SLURM_JOB_ID}.mp4"
mkdir -p "$OUT_DIR"
printf '%s\n' "$PROMPT" > "${OUT%.mp4}.prompt.txt"

ARGS=(--duration "$DURATION" --seed "$SEED" --steps "$STEPS")
[ -n "${ASPECT:-}" ]     && ARGS+=(--aspect "$ASPECT")
[ -n "${IMAGE:-}" ]      && ARGS+=(--image "$IMAGE")
[ -n "${LAST_IMAGE:-}" ] && ARGS+=(--last-image "$LAST_IMAGE")
for ref in ${REFS:-}; do
    ARGS+=(--ref "$ref")
done

cluster_banner
echo "Video      : ${DURATION} s, aspect ${ASPECT:-auto}, seed $SEED, $STEPS steps"
[ -n "${IMAGE:-}${LAST_IMAGE:-}" ] && echo "Keyframes  : ${IMAGE:--} -> ${LAST_IMAGE:--}"
[ -n "${REFS:-}" ] && echo "References : $REFS"
echo "Output     : $OUT"
echo "=========================================="

START=$(date +%s)
python "$TEMPLATES_DIR/repo/models/minimax-h3/generate.py" \
    --model-dir "$TEMPLATES_DIR/models/MiniMax-H3" \
    --prompt-file "${OUT%.mp4}.prompt.txt" \
    --output "$OUT" \
    "${ARGS[@]}"

echo "Done in $(( $(date +%s) - START )) s: $OUT"
