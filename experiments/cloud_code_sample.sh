#!/bin/bash
# The coding agent's price test on the rented card (2026-10-27, user: "test the code domain and give me a
# price estimate"): both steps timed on this card before the whole pass (12,960 requests) is priced:
#   1. Gemma plays a sample of requests (scope, then explanation / found)
#   2. the sample's output vocabulary, then Gemma reads its replies back (the teacher data)
# The sample is the variant pilot_cloud, a trial that is never used again. The machine stops itself at
# the end, on any failure, and after MAX_HOURS whatever happens.
#
#   tmux new-session -d -s run "bash experiments/cloud_code_sample.sh 1 64 16 32"   # 64 train + 32 test, 16 per call, 32 at a time
# Log: /workspace/logs/code_sample.log (GPU memory every 30 s: /workspace/logs/code_sample_gpu.log)

MAX_HOURS=${1:-1}
COUNT=${2:-64}
BATCH=${3:-16}
CHUNK=${4:-32}
VARIANT=pilot_cloud
LOG=/workspace/logs/code_sample.log

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
    >> /workspace/logs/code_sample_gpu.log 2>&1 &

step "Gemma's pass: $COUNT train + $((COUNT / 2)) test requests, $BATCH per call, $CHUNK at a time"
python -u -m library.domain.code_agent.harness "$VARIANT" "$COUNT" "$BATCH" "$CHUNK" || stop_machine "the pass failed"

step "the output vocabulary"
python -u -m library.domain.code_agent.vocabulary "$VARIANT" || stop_machine "the vocabulary failed"

step "teacher data"
python -u -c "import sys; sys.argv = ['code_agent_trial.py', '$VARIANT']; sys.path.insert(0, 'experiments'); \
import code_agent_trial; sys.exit(0 if code_agent_trial.save_teacher_data() else 1)" || stop_machine "the teacher data failed"

stop_machine "done"
