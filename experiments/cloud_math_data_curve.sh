#!/bin/bash
# The math data curve on a rented card (chapter 9 plan #4, 2026-11-07): the three points of
# experiments/math_data_curve.py (6k, 20k, 65k training replies) side by side, one process each.
# Unattended: the machine stops itself at the end, on a failure, and after MAX_HOURS whatever happens
# (the money cap; every run resumes from its last checkpoint, so a stop at the cap loses only the pass
# in progress).
#
# Layout: the network volume at /workspace holds the code (/workspace/preference_destillation), the
# uploaded batch files (/workspace/batches/...) and the runs; the batch files are copied to the
# container disk first (the runs read them 10 times) and linked into the code folder.
#
#   tmux new-session -d -s run "bash experiments/cloud_math_data_curve.sh 9"   # 9 h on a $0.89/h card: $8
# Log: /workspace/logs/data_curve.log

MAX_HOURS=${1:?the money cap in hours: the budget divided by the hourly price of the card}
BATCHES=batches/math_tutor_full_v1/google_gemma-4-12B-it
LOG=/workspace/logs/data_curve.log

. /etc/rp_environment
cd /workspace/preference_destillation
mkdir -p /workspace/logs
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M')"; }
# runpodctl refuses the pod's own key ("Unauthorized"); RunPod's GraphQL API takes it (tested 2026-10-25,
# used at the end of both math loops). Retried, in case the API does not answer.
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

step "packages"
pip install --break-system-packages -q -r requirements.txt || stop_machine "pip failed"
pip uninstall --break-system-packages -y -q torchvision torchaudio

step "copy $(ls /workspace/$BATCHES | wc -l) batch files to the container disk"
mkdir -p /root/$BATCHES
cp -n /workspace/$BATCHES/* /root/$BATCHES/ || stop_machine "the copy failed"
mkdir -p batches/math_tutor_full_v1
[ -L $BATCHES ] || ln -s /root/$BATCHES $BATCHES

step "three students side by side"
for point in 06k 20k 65k; do
    python -u experiments/math_data_curve.py $point > /workspace/logs/data_curve_$point.log 2>&1 &
done
wait

python - <<'EOF' || stop_machine "a student did not finish"
import sys
sys.path.insert(0, ".")
from library.training.training_runner import is_completed
names = [f"math-agent-kl-m-s1-data{point}" for point in ("06k", "20k", "65k")]
missing = [name for name in names if not is_completed(name)]
print("completed:", [name for name in names if name not in missing], "missing:", missing)
sys.exit(1 if missing else 0)
EOF

stop_machine "done"
