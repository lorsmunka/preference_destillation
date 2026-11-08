#!/bin/bash
# The math tutor's first 30k training examples + its whole test set (the same test files as the full runs:
# runs/math_30k_files.json, from split_batch_files) to the network volume math-30k (EU-RO-1), for the
# 30k-example KL/CE runs (thesis 9.1, 2026-11-08). Resumable: every pass lists what the volume already has
# (name and size) and sends only the files missing or incomplete there; up to 50 passes.
#
#   bash experiments/upload_math_30k.sh <host> <port>       (log: runs/math_upload.log)

HOST=${1:?host}
PORT=${2:?port}
STREAMS=${3:-4}
SOURCE=batches/math_tutor_full_v1/google_gemma-4-12B-it
TARGET=/workspace/batches/math_tutor_full_v1_30k/google_gemma-4-12B-it
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -c aes128-gcm@openssh.com -p $PORT root@$HOST"
cd "$(dirname "$0")/.." || exit 1
exec >> runs/math_upload.log 2>&1

FILES=$(python -c "
import json
files = json.load(open('runs/math_30k_files.json'))
print('\n'.join(files['train'] + files['test'] + ['info.json', 'input_vocabulary.json', 'state.json', 'generation.jsonl']))" | tr -d '\r')

for pass_number in $(seq 1 50); do
    $SSH "mkdir -p $TARGET && cd $TARGET && find . -maxdepth 1 -type f -printf '%f %s\n'" > runs/math_upload_remote.txt || { sleep 60; continue; }
    PENDING=()
    for name in $FILES; do
        local_size=$(stat -c %s "$SOURCE/$name")
        remote_size=$(grep -m1 "^$name " runs/math_upload_remote.txt | cut -d' ' -f2)
        [ "$local_size" = "$remote_size" ] || PENDING+=("$SOURCE/$name")
    done
    echo "== pass $pass_number $(date '+%H:%M'): ${#PENDING[@]} files to send"
    [ ${#PENDING[@]} -eq 0 ] && { echo "UPLOAD_DONE"; exit 0; }
    # STREAMS connections at once (one connection was the limit, not the line: the user, 2026-11-08), each taking
    # every STREAMS-th file
    for ((stream = 0; stream < STREAMS; stream++)); do
        (
            share=()
            for ((index = stream; index < ${#PENDING[@]}; index += STREAMS)); do share+=("${PENDING[index]}"); done
            for ((start = 0; start < ${#share[@]}; start += 10)); do
                scp -q -o ConnectTimeout=30 -o ServerAliveInterval=30 -c aes128-gcm@openssh.com -P "$PORT" \
                    "${share[@]:start:10}" "root@$HOST:$TARGET/" || echo "  stream $stream: scp failed, retried next pass"
            done
            echo "  stream $stream done $(date '+%H:%M')"
        ) &
    done
    wait
done
echo "UPLOAD_NOT_DONE after 50 passes"
