#!/bin/bash
# =============================================================================
# vertical_20s.sh -- render a 20 s vertical (1088x1920) LTX-2.5 video with sound
# from a prompt on the command line. Submits generate.sbatch, follows the job
# log until the job ends, and prints the path of the video.
#
#   bash vertical_20s.sh "A street musician in Vienna plays the violin, then says in German: \"Danke fürs Zuhören.\""
#   SEED=7 bash vertical_20s.sh "..."        # another take of the same prompt
#
# Takes about 5 minutes. Ctrl-C stops following the log; the job keeps running.
# =============================================================================
set -euo pipefail

if [ $# -ne 1 ] || [[ "$1" != *" "* ]]; then
    echo "usage: $0 \"<prompt text>\"" >&2
    exit 2
fi

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 481 frames = 20 s at 24 fps (frame counts must be 8k+1). 1088x1920 is
# vertical 1080p-class; both sides must be divisible by 64.
JOBID=$(NUM_FRAMES=481 WIDTH=1088 HEIGHT=1920 SEED="${SEED:-42}" \
        sbatch --parsable "$HERE/generate.sbatch" "$1")
LOG="$TEMPLATES_DIR/logs/ltx25_generate_${JOBID}"
OUT="$TEMPLATES_DIR/output/ltx-2.5/ltx25_${JOBID}.mp4"
echo "Submitted job $JOBID. Waiting for it to start ..."

# Slurm creates the log files when the job starts.
until [ -f "$LOG.out" ] || ! squeue -h -j "$JOBID" 2>/dev/null | grep -q .; do
    sleep 2
done
tail -n +1 -F "$LOG.out" "$LOG.err" 2>/dev/null &
TAIL_PID=$!
trap 'kill $TAIL_PID 2>/dev/null || true' EXIT

while squeue -h -j "$JOBID" 2>/dev/null | grep -q .; do
    sleep 5
done
sleep 2   # let tail print the last lines
kill $TAIL_PID 2>/dev/null || true

echo ""
if grep -q "Done in" "$LOG.out" 2>/dev/null && [ -s "$OUT" ]; then
    echo "Video: $OUT"
else
    echo "Job $JOBID ended without a video; see $LOG.err" >&2
    exit 1
fi
