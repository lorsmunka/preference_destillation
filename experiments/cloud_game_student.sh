#!/bin/bash
# On the scoring card (2026-11-08): replaces cloud_game.sh's end of the scoring step (which stops the machine) with:
# wait for the scoring, check it completed, train the game agent's first student (experiments/game_student.py),
# then stop the machine. Stops itself on any failure and after <hours> whatever happens (the money cap).
#
#   PYTHON=/workspace/venv/bin/python bash experiments/cloud_game_student.sh 11     (log: /workspace/logs/game_student.log)

MAX_HOURS=${1:?the money cap in hours}
PY=${PYTHON:-python}
LOG=/workspace/logs/game_student.log
CAP_PID=/workspace/logs/money_cap.pid
INFO=batches/game_dsl/google_gemma-4-12B-it/info.json

. /etc/rp_environment
export HF_HOME=/workspace/hf
cd /workspace/pd
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M')"; }
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

# cloud_game.sh would stop the machine when the scoring ends: end it (the scoring itself runs on) and its money cap
pkill -f '[c]loud_game.sh score'
[ -f "$CAP_PID" ] && kill "$(cat "$CAP_PID")" 2>/dev/null
( sleep "${MAX_HOURS}h"; stop_machine "the money cap ($MAX_HOURS h)" ) &
echo $! > "$CAP_PID"

step "waiting for the scoring (money cap $MAX_HOURS h)"
while pgrep -f '[g]ame_gemma_run.py score' > /dev/null; do sleep 60; done
grep -q '"status": "completed"' "$INFO" || stop_machine "the scoring is not complete"
step "scoring complete; the student"
"$PY" -u experiments/game_student.py || stop_machine "the student failed"
stop_machine "done"
