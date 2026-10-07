#!/bin/bash
# =============================================================================
# query.sh -- one chat completion against a running glm53-serve job.
#
#   bash models/glm-5.3/query.sh                         # default prompt
#   bash models/glm-5.3/query.sh "Explain MoE routing in two sentences."
#   REASONING_EFFORT=low bash models/glm-5.3/query.sh "..."   # low|high; unset = max
#   SERVE_JOBID=<jobid> bash models/glm-5.3/query.sh ... # pick among several
#   HOST=<node> PORT=8000 bash models/glm-5.3/query.sh ...
#
# In a multi-node job only the head runs the HTTP server; the others are
# --headless. The head is the job's batch host -- NOT the first name in
# squeue's %N, which is sorted: a 2-node job on worker-b300-[0-1] had its head,
# and its endpoint, on worker-b300-1.
# =============================================================================
set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-${DEMO_DIR:-/mnt/data/slurm-llm-templates}}"
PROMPT=${1:-"In one sentence: what is a mixture-of-experts model?"}
PORT=${PORT:-8000}
MODEL=${MODEL:-}             # default: whatever the server serves
REASONING_EFFORT=${REASONING_EFFORT:-}   # low|high; empty = model default (max)
MAX_TOKENS=${MAX_TOKENS:-4096}

if [ -z "${HOST:-}" ]; then
    if [ -z "${SERVE_JOBID:-}" ]; then
        mapfile -t JOBS < <(squeue --noheader -n glm53-serve -t RUNNING -o "%i %N")
        if [ "${#JOBS[@]}" -gt 1 ]; then
            echo "Several glm53-serve jobs are running; set SERVE_JOBID or HOST:" >&2
            printf '  %s\n' "${JOBS[@]}" >&2
            exit 1
        fi
        SERVE_JOBID=$(printf '%s\n' "${JOBS[0]:-}" | awk '{print $1}')
    fi
    [ -n "$SERVE_JOBID" ] || {
        echo "No running glm53-serve job. Start one with:" >&2
        echo "  sbatch $TEMPLATES_DIR/repo/models/glm-5.3/serve.sbatch" >&2
        exit 1
    }
    HOST=$(squeue --noheader -j "$SERVE_JOBID" -t RUNNING -O BatchHost | tr -d ' ')
    [ -n "$HOST" ] || { echo "Job $SERVE_JOBID is not running." >&2; exit 1; }
fi
URL="http://${HOST}:${PORT}"

# Loading 704 GiB and capturing CUDA graphs takes a while; wait for the
# endpoint rather than failing on a server that is still starting.
if ! curl -sf -m 5 "$URL/v1/models" >/dev/null 2>&1; then
    echo "Waiting for $URL (tail -f $TEMPLATES_DIR/logs/glm53_serve_<JOBID>.out)..." >&2
    for _ in $(seq 1 180); do
        sleep 10
        curl -sf -m 5 "$URL/v1/models" >/dev/null 2>&1 && break
    done
    curl -sf -m 5 "$URL/v1/models" >/dev/null 2>&1 || { echo "$URL never answered" >&2; exit 1; }
fi

if [ -z "$MODEL" ]; then
    MODEL=$(curl -sf -m 5 "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
fi
echo "Server : $URL   model: $MODEL" >&2

BODY=$(python3 - "$MODEL" "$PROMPT" "$REASONING_EFFORT" "$MAX_TOKENS" <<'PYEOF'
import json, sys
model, prompt, effort, max_tokens = sys.argv[1:5]
body = {
    "model": model,
    "max_tokens": int(max_tokens),
    "messages": [{"role": "user", "content": prompt}],
}
# GLM-5.3 always reasons; its chat template has no enable_thinking switch, and
# passing enable_thinking=false makes vLLM put the reasoning into the answer.
# reasoning_effort (low|high, default max) is the control it does have.
if effort:
    body["reasoning_effort"] = effort
print(json.dumps(body))
PYEOF
)

START=$(date +%s.%N)
RESPONSE=$(curl -sS --fail-with-body -m 600 "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" --data-binary "$BODY") || {
    echo "Request failed: $RESPONSE" >&2
    exit 1
}
END=$(date +%s.%N)

echo "$RESPONSE" | python3 -c "
import json, sys
r = json.load(sys.stdin)
msg, usage = r['choices'][0]['message'], r.get('usage', {})
reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
if reasoning:
    print('--- reasoning (%d chars) ---' % len(reasoning))
    print(reasoning.strip()[:2000])
    print('--- answer ---')
print((msg.get('content') or '').strip())
n, dt = usage.get('completion_tokens', 0), $END - $START
print(f'\n[{n} completion tokens in {dt:.1f}s = {n/dt:.1f} tok/s, finish={r[\"choices\"][0][\"finish_reason\"]}]')
"
