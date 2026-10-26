#!/bin/bash
# The math tutor's second loop on the rented card (2026-10-26, user: "let's first make more data", a $55
# budget): more data only, no students (they train at home). Unattended:
#   1. Gemma plays every problem not played yet (the pass skips the played ones)
#   2. Gemma reads its new replies back: the new records (train, then test) are added after the corpus's
#      records and scored into new batch files after the others; the vocabulary stays as it is
#      (library/domain/math_tutor/full.py)
# The machine stops itself at the end, on any failure, and after MAX_HOURS whatever happens (the
# money cap: every step resumes, so a stop at the cap loses nothing that was saved).
#
#   tmux new-session -d -s run "bash experiments/cloud_math_more.sh 29"        # 29 h on the A100 ($1.79/h): ~$52
#   tmux new-session -d -s run "bash experiments/cloud_math_more.sh 21 32"     # another card: hours = budget / price
# Log: /workspace/logs/math_more.log

MAX_HOURS=${1:?the money cap in hours: the budget divided by the hourly price of the card}
BATCH=${2:-48}
CONVERSATIONS=library/domain/math_tutor/items/conversations/full_v1.jsonl
TEACHER_DATA_INFO=batches/math_tutor_full_v1/google_gemma-4-12B-it/info.json
LOG=/workspace/logs/math_more.log

. /etc/rp_environment
cd /workspace/preference_destillation
mkdir -p /workspace/logs
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M')"; }
conversation_count() { [ -f "$CONVERSATIONS" ] && wc -l < "$CONVERSATIONS" || echo 0; }
# runpodctl refuses the pod's own key ("Unauthorized"); RunPod's GraphQL API takes it (tested on a
# throwaway pod, 2026-10-25; used at the end of the first loop). Retried, in case the API does not answer.
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

step "Gemma's conversations: $(conversation_count) played so far, every problem left, $BATCH replies per call, cap $MAX_HOURS h"
python -u -m library.domain.math_tutor.full conversations "$BATCH" || stop_machine "the pass failed"

step "teacher data of the new replies ($(conversation_count) conversations)"
python -u experiments/math_full.py v1 none || stop_machine "the teacher data failed"
grep -q '"status": "completed"' "$TEACHER_DATA_INFO" || stop_machine "the teacher data is not complete (stopped before the end)"

stop_machine "done"
