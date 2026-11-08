#!/bin/bash
# The KL vs CE runs at 30k examples (experiments/kl_ce_30k.py) on a rented card, one process per run, all at once;
# the machine stops itself after <hours> whatever happens (the money cap). When every run has ended, ALL_DONE
# appears in /workspace/logs: the runs are then downloaded and the pod terminated.
#
#   bash experiments/cloud_klce.sh reddit 3 <run name> <run name> ...       (logs: /workspace/logs/<run name>.log)

KEY=${1:?reddit or math}
MAX_HOURS=${2:?the money cap in hours}
shift 2
. /etc/rp_environment
cd /workspace/pd
mkdir -p /workspace/logs
stop_machine() {
    echo "== $1: stopping the machine $(date -u '+%H:%M')" >> /workspace/logs/klce.log
    for attempt in 1 2 3 4 5; do
        curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
            https://api.runpod.io/graphql \
            -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" \
            >> /workspace/logs/klce.log
        sleep 30
    done
    exit
}
( sleep "${MAX_HOURS}h"; stop_machine "the money cap ($MAX_HOURS h)" ) &
pids=()
for name in "$@"; do
    python -u experiments/kl_ce_30k.py "$KEY" "$name" > "/workspace/logs/$name.log" 2>&1 &
    pids+=($!)
done
wait "${pids[@]}"       # the runs (not the money cap); the container disk holds the results: no stop at the end,
                        # they are downloaded, then the pod is terminated
for name in "$@"; do grep -q "completed" "/workspace/logs/$name.log" && echo "$name completed" >> /workspace/logs/klce.log; done
echo "== all runs ended $(date -u '+%H:%M')" >> /workspace/logs/klce.log
touch /workspace/logs/ALL_DONE
