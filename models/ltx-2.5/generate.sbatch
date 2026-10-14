#!/bin/bash
# =============================================================================
# generate.sbatch -- one LTX-2.5 video with sound per job, on one node.
#
#   sbatch generate.sbatch prompts/en-interview.txt     # prompt from a file
#   sbatch generate.sbatch "A red fox trots through fresh snow at dawn ..."
#
# Writes $TEMPLATES_DIR/output/ltx-2.5/ltx25_<JOBID>.mp4 (H.264 + AAC) and a
# copy of the prompt next to it, then exits.
#
# KNOBS (environment, all optional):
#   PIPELINE     distilled (default): fast, 8+4 steps.
#                dfr: production quality -- same distilled transformer plus a
#                detailing IC-LoRA pass; slower.
#   NUM_FRAMES   default 121 (5 s at 24 fps). Must be 8k+1: 121, 241, 481, ...
#                "auto" lets the duration head pick 1-20 s from the prompt.
#                The model is made for up to 20 s (481); 721 (30 s) still
#                spoke all 79 words of a German monologue but drifted in the
#                last second.
#   WIDTH/HEIGHT default 1536x1024; both divisible by 64. 1920x1088 is 1080p-
#                class, 1088x1920 portrait.
#   FRAME_RATE   default 24
#   SEED         default 42
#   IMAGE        optional path to a first frame (image-to-video)
#   EXTRA_ARGS   appended verbatim to the ltx_pipelines command. Not
#                "--chunked" for clips with speech: every window re-reads the
#                whole prompt, so the speaker starts the dialogue over at each
#                window (30 s chunked: WER 0.85 against 0.01 in one pass).
#
# GPUs: one GPU per video, bf16, eight jobs side by side on a node. Measured on
# B300, the whole job for a 5 s clip at 1536x1024 takes about 3.5 min with
# eight running at once; 30 s takes about 5.5 min.
#
# With more GPUs (sbatch --gpus-per-node=8 --exclusive --mem=0) the job runs
# the upstream *_mgpu runners instead, which split every denoising step across
# the GPUs. On this cluster that is not faster per job: spawning eight
# workers that each load the full 70 GB of weights from shared storage takes
# longer than the split saves (about 5.5 min for 5 s). The runners also cast
# the transformer to FP8 and cap a clip at 32768 video tokens, i.e. 161 frames
# at 1536x1024 and 121 at 1920x1088; longer requests fail with "Total video
# token count ... exceeds attention_manager max_tokens". LTX runs on one node
# only.
# =============================================================================

#SBATCH --job-name=ltx25-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
# DefMemPerNode is UNLIMITED here: without --mem a job claims all 2.4 TB of the
# node and the other seven GPUs sit idle. 250G per job leaves room for eight;
# it held for a 30 s clip and for the DFR pipeline.
#SBATCH --mem=250G
#SBATCH --partition=main
# Both partitions have TIMELIMIT=infinite; a job that hangs in NCCL setup would
# hold the node forever.
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/slurm-llm-templates/logs/ltx25_generate_%j.out
#SBATCH --error=/mnt/data/slurm-llm-templates/logs/ltx25_generate_%j.err

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
# shellcheck source=../../cluster/env.sh
source "$TEMPLATES_DIR/repo/cluster/env.sh"
cluster_setup venv-ltx
# colour-science warns on import that matplotlib is missing, once per GPU
# worker. Its plotting API is not used.
export PYTHONWARNINGS='ignore:"Matplotlib" related API features are not available'

if [ $# -ne 1 ]; then
    echo "usage: sbatch generate.sbatch <prompt-file | \"prompt text\">" >&2
    exit 2
fi
# sbatch copies the script to a spool directory, so a relative prompt path
# resolves against the submit directory, which is where the job starts.
# The file must be visible from the compute node: a prompt file under the login
# node's /tmp is not, and a single word with no file behind it is a path that
# did not resolve, not a prompt. Using it as the prompt renders the path.
if [ -f "$1" ]; then
    PROMPT="$(cat "$1")"
elif [[ "$1" != *[[:space:]]* ]]; then
    echo "Prompt file '$1' not found on $(hostname). Put it on shared storage" \
         "(e.g. under \$TEMPLATES_DIR or your home directory)." >&2
    exit 2
else
    PROMPT="$1"
fi

PIPELINE="${PIPELINE:-distilled}"
NUM_FRAMES="${NUM_FRAMES:-121}"
WIDTH="${WIDTH:-1536}"
HEIGHT="${HEIGHT:-1024}"
FRAME_RATE="${FRAME_RATE:-24}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
GPUS="${SLURM_GPUS_ON_NODE:-1}"

M="$TEMPLATES_DIR/models/LTX-2.5"
OUT_DIR="$TEMPLATES_DIR/output/ltx-2.5"
OUT="$OUT_DIR/ltx25_${SLURM_JOB_ID}.mp4"
mkdir -p "$OUT_DIR"
printf '%s\n' "$PROMPT" > "${OUT%.mp4}.prompt.txt"

case "$PIPELINE" in
    distilled) MODULE=distilled
               PIPELINE_ARGS=() ;;
    dfr)       MODULE=dfr_pipeline
               PIPELINE_ARGS=(
                   --detailing-lora "$M/loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
                   --temporal-upsampler-path "$M/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
               ) ;;
    *) echo "PIPELINE must be distilled or dfr (got '$PIPELINE')" >&2; exit 2 ;;
esac
if [ "$GPUS" -gt 1 ]; then
    case "$MODULE" in
        distilled)    MODULE=distilled_mgpu ;;
        dfr_pipeline) MODULE=dfr_mgpu ;;
    esac
fi

FRAME_ARGS=()
[ "$NUM_FRAMES" != auto ] && FRAME_ARGS=(--num-frames "$NUM_FRAMES")
IMAGE_ARGS=()
[ -n "${IMAGE:-}" ] && IMAGE_ARGS=(--image "$IMAGE" 0 1.0)

cluster_banner
echo "Pipeline   : ltx_pipelines.$MODULE on $GPUS GPU(s)"
echo "Video      : ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames at ${FRAME_RATE} fps, seed $SEED"
echo "Output     : $OUT"
echo "=========================================="

START=$(date +%s)
# shellcheck disable=SC2086  # EXTRA_ARGS is meant to word-split
python -m "ltx_pipelines.$MODULE" \
    --transformer-path       "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path      "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path         "$M/vae/ltx-2.5-video-vae-bf16.safetensors" \
    --audio-vae-path         "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --duration-head-path     "$M/model_patches/ltx-2.5-duration-head-bf16.safetensors" \
    --spatial-upsampler-path "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    "${PIPELINE_ARGS[@]}" "${FRAME_ARGS[@]}" "${IMAGE_ARGS[@]}" \
    --width "$WIDTH" --height "$HEIGHT" --frame-rate "$FRAME_RATE" --seed "$SEED" \
    --prompt "$PROMPT" \
    --output-path "$OUT" \
    $EXTRA_ARGS

echo "Done in $(( $(date +%s) - START )) s: $OUT"
