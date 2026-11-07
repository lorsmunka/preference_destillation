#!/bin/bash
# The game-language agent's Gemma run on the rented card (2026-11-07; experiments/game_gemma_run.py). Two steps,
# the check at home between them (the app's compiler is a Windows program):
#   bash experiments/cloud_game.sh write 7    pass 1: Gemma writes every block (~4.2 h for 10,378); the card then
#                                             waits for the check at home (the cap still runs)
#   bash experiments/cloud_game.sh score 3    pass 2: the kept blocks scored (~1.2 h), after corpus.jsonl and
#                                             output_tokens.json are uploaded; the machine then stops
# The machine stops itself after <hours> whatever happens (the money cap; a new step replaces the old cap), on a
# failure, and after pass 2. Everything lives on the pod's persistent disk (/workspace), which a stop keeps.
# Log: /workspace/logs/game_<step>.log

STEP=${1:?write or score}
MAX_HOURS=${2:?the money cap in hours}
LOG=/workspace/logs/game_$STEP.log
CAP_PID=/workspace/logs/money_cap.pid

. /etc/rp_environment
export HF_HOME=/workspace/hf
cd /workspace/pd
mkdir -p /workspace/logs
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M')"; }
# runpodctl refuses the pod's own key ("Unauthorized"); RunPod's GraphQL API takes it (tested 2026-10-25, used by
# experiments/cloud_math_more.sh). Retried, in case the API does not answer.
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

[ -f "$CAP_PID" ] && kill "$(cat "$CAP_PID")" 2>/dev/null
( sleep "${MAX_HOURS}h"; stop_machine "the money cap ($MAX_HOURS h)" ) &
echo $! > "$CAP_PID"

step "$STEP (money cap $MAX_HOURS h)"
python -u experiments/game_gemma_run.py "$STEP" || stop_machine "$STEP failed"
step "$STEP done"
[ "$STEP" = score ] && stop_machine "done"
exit 0
