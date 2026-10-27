#!/bin/bash
# The coding agent's data on the rented card, unattended (2026-10-27): no students (they train at home).
#   1. Gemma plays every request not played yet (scope, then explanation / found)
#   2. the output vocabulary of its replies, then Gemma reads its replies back (the teacher data)
# The machine stops itself at the end, on any failure, and after MAX_HOURS whatever happens (the
# money cap: every step resumes, so a stop at the cap loses only the chunk in progress).
#
#   tmux new-session -d -s run "bash experiments/cloud_code.sh 21"                    # all 12,960 requests (v1), 12 per call, 48 at a time
#   tmux new-session -d -s run "bash experiments/cloud_code.sh 1 16 32 pilot_cloud 64" # the price test of 2026-10-27: 64 train + 32 test
# Log: /workspace/logs/code_<variant>.log (GPU memory every 30 s: code_<variant>_gpu.log)

MAX_HOURS=${1:?the money cap in hours: the budget divided by the hourly price of the card}
BATCH=${2:-12}
CHUNK=${3:-48}
VARIANT=${4:-v1}
COUNT=${5:--}
LOG=/workspace/logs/code_$VARIANT.log
TEACHER_DATA_INFO=batches/$([ "$VARIANT" = v1 ] && echo code_agent || echo "code_agent_$VARIANT")/google_gemma-4-12B-it/info.json

. /etc/rp_environment
cd /workspace/preference_destillation
mkdir -p /workspace/logs
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M:%S')"; }
# the same stop as cloud_math_more.sh (RunPod's GraphQL API with the pod's own key, tested 2026-10-25)
stop_machine() {
    step "$1: stopping the machine"
    for attempt in 1 2 3 4 5; do
        reply=$(curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
                https://api.runpod.io/graphql \
                -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}")
        echo "podStop attempt $attempt: $reply"
        sleep 30
    done
    exit
}

( sleep "${MAX_HOURS}h"; stop_machine "the money cap ($MAX_HOURS h)" ) &
( while true; do echo "$(date -u '+%H:%M:%S') $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"; sleep 30; done ) \
    >> "/workspace/logs/code_${VARIANT}_gpu.log" 2>&1 &

step "Gemma's pass ($VARIANT, requests: $COUNT), $BATCH per call, $CHUNK at a time, cap $MAX_HOURS h"
python -u -m library.domain.code_agent.harness "$VARIANT" "$COUNT" "$BATCH" "$CHUNK" || stop_machine "the pass failed"
played=$(wc -l < "library/domain/code_agent/conversations/$VARIANT.jsonl")
expected=$([ "$COUNT" = - ] && wc -l < library/domain/code_agent/items.jsonl || echo $((COUNT + COUNT / 2)))
[ "$played" -ge "$expected" ] || stop_machine "the pass stopped before the end ($played of $expected played)"

step "the output vocabulary"
python -u -m library.domain.code_agent.vocabulary "$VARIANT" || stop_machine "the vocabulary failed"

step "teacher data"
python -u -c "import sys; sys.argv = ['code_agent_trial.py', '$VARIANT']; sys.path.insert(0, 'experiments'); \
import code_agent_trial; sys.exit(0 if code_agent_trial.save_teacher_data() else 1)" || stop_machine "the teacher data failed"
grep -q '"status": "completed"' "$TEACHER_DATA_INFO" || stop_machine "the teacher data is not complete (stopped before the end)"

stop_machine "done"
