#!/bin/bash
# The game-language agent v2 overnight, on each rented card, with no one at home (the user, 2026-11-09: "PC off after
# ~23:45"): score this card's share of the corpus, then either hand the batches to the trainer cards (a scorer) or
# gather every share and train students (a trainer), push the students to the receiver, stop.
#
#   bash experiments/cloud_game_v2.sh <this card's name> "<student sizes, empty for a scorer>"
#
# Files the run needs (put there from home before it starts scoring):
#   /workspace/pd                       the code, with library/domain/game_dsl/corpus_v2.jsonl = this card's share
#                                       and output_tokens_v2.json (the same vocabulary on every card)
#   /workspace/cards.txt                one line per card: <name> <ip> <ssh port> <role: scorer|trainer>
#   /workspace/receiver.txt             <ip> <ssh port>: the CPU pod with the math-30k volume at /vol
#   /root/.ssh/card_key                 the key the cards use to reach each other and the receiver
#   /workspace/GO                       written last: scoring starts when it exists
# Every card stops itself: after its work, on a failure, and at the money cap whatever happens.

NAME=${1:?card name}
SIZES=${2:-}
MAX_HOURS=${MAX_HOURS:-11}
PY=${PYTHON:-python}      # the writing cards score from a venv (torch 2.8, transformers 5.17): vLLM brought its own torch
. /etc/rp_environment 2>/dev/null
export HF_HOME=/root/hf
LOG=/workspace/logs/v2_$NAME.log
BATCHES=/workspace/pd/batches/game_dsl_v2/google_gemma-4-12B-it
SSH="ssh -i /root/.ssh/card_key -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
mkdir -p /workspace/logs
exec >> "$LOG" 2>&1

step() { echo "== $1 $(date -u '+%Y-%m-%d %H:%M:%S')"; }
stop_machine() {
    step "$1: stopping the machine"
    [ -n "$BACKUP_PID" ] && { step "waiting for the backup push"; wait "$BACKUP_PID"; }
    [ -n "$DRY_RUN" ] && { echo "(dry run: no podStop)"; exit; }
    if [ -n "$SMOKE" ]; then     # the end-to-end test: only check that the API takes this card's key
        curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
             https://api.runpod.io/graphql -d '{"query": "query { myself { id } }"}'
        echo; echo "(smoke test: the key works if an id is printed above; no podStop)"
        exit
    fi
    for attempt in 1 2 3 4 5; do
        reply=$(curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
                https://api.runpod.io/graphql \
                -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}")
        echo "podStop attempt $attempt: $reply"
        sleep 30
    done
    exit
}
card_field() { awk -v name="$1" -v field="$2" '$1 == name { print $field }' /workspace/cards.txt; }
on_card() { $SSH -p "$(card_field "$1" 3)" "root@$(card_field "$1" 2)" "$2"; }

( sleep "${MAX_HOURS}h"; stop_machine "the money cap ($MAX_HOURS h)" ) &

step "setup ($NAME, sizes: ${SIZES:-none})"
if [ -z "$DRY_RUN" ]; then
    [ -n "$PYTHON" ] || pip install --break-system-packages -q 'transformers==5.17.0' accelerate matplotlib tqdm pandas protobuf \
        || stop_machine "pip failed"
    "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-12B-it'))" \
        || stop_machine "the Gemma download failed"
fi
step "waiting for GO"
while [ ! -f /workspace/GO ]; do sleep 30; done

step "scoring $(wc -l < /workspace/pd/library/domain/game_dsl/corpus_v2.jsonl) records"
cd /workspace/pd
if [ -n "$DRY_RUN" ]; then
    mkdir -p "$BATCHES" && for i in 0 1 2; do echo "$NAME $i" > "$BATCHES/batch_$i.npz"; done
    echo '{"status": "completed"}' > "$BATCHES/info.json"
else
    "$PY" -u experiments/game_v2_run.py score 0 8 || "$PY" -u experiments/game_v2_run.py score 0 8 \
        || stop_machine "the scoring failed twice"
fi
grep -q '"status": "completed"' "$BATCHES/info.json" || stop_machine "the scoring did not complete"
touch /workspace/SCORE_DONE
step "scoring done: $(ls "$BATCHES" | grep -c npz) batch files"

# the backup: this card's share to the game-v2 network volume (the keeper pod), in the background; every stop waits
# for it, so no share exists on one card's disk alone
read -r receiver_ip receiver_port < /workspace/receiver.txt
( tar -C "$BATCHES" -cf - . | $SSH -p "$receiver_port" "root@$receiver_ip" \
      "mkdir -p /vol/game_v2_batches/$NAME && tar -C /vol/game_v2_batches/$NAME -xf -" \
  && step "this card's share is backed up on the volume" || step "the backup push FAILED (the share stays on this disk)" ) &
BACKUP_PID=$!

if [ -z "$SIZES" ]; then
    # a scorer: wait until every trainer has copied this card's batches (or 3 h), then stop
    trainers=$(awk '$4 == "trainer" { print $1 }' /workspace/cards.txt)
    for wait in $(seq 360); do
        missing=0
        for trainer in $trainers; do [ -f "/workspace/PULLED_$trainer" ] || missing=1; done
        [ $missing = 0 ] && stop_machine "every trainer has the batches"
        sleep 30
    done
    stop_machine "the trainers did not come for the batches in 3 h"
fi

# a trainer: every other card's batches, as each card finishes. A card that has not finished 90 min after this one
# (or cannot be copied) is left out: the students then train on the shares that did arrive.
mkdir -p /workspace/shards
deadline=$(( $(date +%s) + 90 * 60 ))
for card in $(awk '{ print $1 }' /workspace/cards.txt); do
    [ "$card" = "$NAME" ] && continue
    while [ "$(date +%s)" -lt "$deadline" ]; do on_card "$card" "test -f /workspace/SCORE_DONE" && break; sleep 30; done
    if ! on_card "$card" "test -f /workspace/SCORE_DONE"; then
        step "$card did not finish in time: left out"
        continue
    fi
    step "copying $card's batches"
    mkdir -p "/workspace/shards/$card"
    # a trainer that has merged already keeps its own batches in shards/<its name> (symlinks to every share in
    # their place); the 2026-11-10 run fixed this on the cards with an ssh forced command instead
    if on_card "$card" "if [ -d /workspace/shards/$card ]; then tar -C /workspace/shards/$card -cf - .; else tar -C $BATCHES -cf - .; fi" \
            | tar -C "/workspace/shards/$card" -xf -; then
        on_card "$card" "touch /workspace/PULLED_$NAME"
    else
        step "copying $card's batches failed: left out"
        rm -rf "/workspace/shards/$card"
    fi
done

step "merging the batches (cards in cards.txt order; each card's test records sit at its tail)"
mv "$BATCHES" "/workspace/shards/$NAME"
mkdir -p "$BATCHES"
index=0
for card in $(awk '{ print $1 }' /workspace/cards.txt); do
    [ -d "/workspace/shards/$card" ] || continue
    for file in $(ls "/workspace/shards/$card" | grep -E '^batch_[0-9]+\.npz$' | sort -t_ -k2 -n); do
        ln -s "/workspace/shards/$card/$file" "$BATCHES/batch_$index.npz"
        index=$((index + 1))
    done
done
cp "/workspace/shards/$NAME/info.json" "$BATCHES/info.json"
step "$index batch files merged"

step "the input vocabulary (once, before the students)"
if [ -z "$DRY_RUN" ]; then
    "$PY" -c "import sys; sys.path.insert(0, '.'); from library.domain.game_dsl import GameDslV2Domain; \
from library.shared.build_input_vocabulary import build_input_vocabulary; \
build_input_vocabulary(GameDslV2Domain(), 'google/gemma-4-12B-it')" || stop_machine "the input vocabulary failed"
fi
step "training: $SIZES"
pids=""
for size in $SIZES; do
    if [ -n "$DRY_RUN" ]; then
        (mkdir -p "/workspace/pd/runs/game-agent-v2-$size-${V2_TAG:-s1}" \
            && ls -l "$BATCHES" > "/workspace/pd/runs/game-agent-v2-$size-${V2_TAG:-s1}/merged.txt") &
    else
        "$PY" -u experiments/game_v2_run.py student "$size" > "/workspace/logs/student_$size.log" 2>&1 &
    fi
    pids="$pids $!"
done
for pid in $pids; do wait "$pid"; done
step "training done"

read -r receiver_ip receiver_port < /workspace/receiver.txt
for size in $SIZES; do
    run=game-agent-v2-$size-${V2_TAG:-s1}
    tar -C /workspace/pd/runs -cf - "$run" | $SSH -p "$receiver_port" "root@$receiver_ip" \
        "mkdir -p /vol/game_v2_students && tar -C /vol/game_v2_students -xf -" \
        && step "$run pushed to the receiver" || step "pushing $run failed (it stays on this card's disk)"
done
cp /workspace/logs/*.log /workspace/pd/runs/ 2>/dev/null
tar -C /workspace/logs -cf - . | $SSH -p "$receiver_port" "root@$receiver_ip" \
    "mkdir -p /vol/game_v2_students/logs_$NAME && tar -C /vol/game_v2_students/logs_$NAME -xf -"
stop_machine "done"
