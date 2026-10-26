#!/bin/bash
# The math tutor's first loop on the rented card (2026-10-25, thesis_work/cloud_runbook.md), unattended:
#   1. Gemma plays problems until PROBLEMS conversations are in full_v1.jsonl (then a graceful stop;
#      a later run of the pass resumes in the same seeded order)
#   2. the output vocabulary, from the replies
#   3. the teacher data (Gemma reads its replies back)
#   4. five students side by side on the one card (one student left it ~75% idle; 6 CPU threads each, 32 cores):
#      kd at s (~6M), l (~53M), xs (~2.6M), m (~16M), and ce at s; then their report
# The machine stops itself at the end, and on any failure, so no credit burns idle.
#
#   tmux new-session -d -s run "bash experiments/cloud_math_first.sh"            # 10,000 problems, 48 replies per call (80 GB)
#   tmux new-session -d -s run "bash experiments/cloud_math_first.sh 10000 16"   # a smaller card
# Log: /workspace/logs/math_first.log

PROBLEMS=${1:-10000}
BATCH=${2:-48}
STUDENTS="kd kd-l kd-xs kd-m ce"
CONVERSATIONS=library/domain/math_tutor/items/conversations/full_v1.jsonl
TEACHER_DATA_INFO=batches/math_tutor_full_v1/google_gemma-4-12B-it/info.json
LOG=/workspace/logs/math_first.log

. /etc/rp_environment
cd /workspace/preference_destillation
mkdir -p /workspace/logs
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M')"; }
conversation_count() { [ -f "$CONVERSATIONS" ] && wc -l < "$CONVERSATIONS" || echo 0; }
# runpodctl refuses the pod's own key ("Unauthorized"); RunPod's GraphQL API takes it (tested on a
# throwaway pod, 2026-10-25). Retried, in case the API does not answer.
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

step "Gemma's conversations: $(conversation_count) of $PROBLEMS played, $BATCH replies per call"
if [ "$(conversation_count)" -lt "$PROBLEMS" ]; then
    python -u -m library.domain.math_tutor.full conversations "$BATCH" &
    pass_id=$!
    while kill -0 "$pass_id" 2>/dev/null; do
        [ "$(conversation_count)" -ge "$PROBLEMS" ] && touch STOP
        sleep 60
    done
    wait "$pass_id"
    rm -f STOP
fi
[ "$(conversation_count)" -ge "$PROBLEMS" ] || stop_machine "the pass ended with only $(conversation_count) conversations"

step "output vocabulary"
python -u -m library.domain.math_tutor.agentic || stop_machine "the vocabulary failed"

step "teacher data"
python -u experiments/math_full.py v1 none || stop_machine "the teacher data failed"
grep -q '"status": "completed"' "$TEACHER_DATA_INFO" || stop_machine "no teacher data (stopped before the end)"

step "the students side by side: $STUDENTS (logs in /workspace/logs/students)"
mkdir -p /workspace/logs/students
for student in $STUDENTS; do
    OMP_NUM_THREADS=6 python -u experiments/math_full.py v1 "$student" > "/workspace/logs/students/$student.log" 2>&1 &
done
wait

step "the students' report (trains any student still missing first)"
python -u experiments/math_full.py v1 "${STUDENTS// /,}" || stop_machine "math_full.py failed"
ls experiments/results/math_full.csv > /dev/null || stop_machine "no report (stopped before the end)"

stop_machine "done"
