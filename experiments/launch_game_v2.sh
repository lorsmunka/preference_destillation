#!/bin/bash
# The game-language agent v2's launch from home (2026-11-09, ~23:00, after Gemma's 4 writing rounds): the last
# replies home, the check (the app's compiler runs here), the vocabulary and corpus, the corpus in 5 shares onto the
# cards (cards.txt), every card checked, a backup of the home data onto the game-v2 volume, then GO on every card.
#
#   bash experiments/launch_game_v2.sh <folder with cards.txt, keeper.txt, card_key>
set -e
KIT=${1:?folder with cards.txt, keeper.txt and card_key}
cd /d/preference_destillation
GAME=library/domain/game_dsl
SSHOPTS="-o ConnectTimeout=20 -o StrictHostKeyChecking=no"

echo "== the last replies (the writing cards keep theirs as well)"
for round in "" 2 3 4; do
    for spec in "a 194.68.245.113 22111" "b 194.68.245.220 22109"; do
        set -- $spec
        remote=/workspace/game_v2_full$round/vllm_written.jsonl
        local_file=runs/game_v2/full$round/written_$1.jsonl
        scp -q $SSHOPTS -P "$3" "root@$2:$remote" "$local_file.new" < /dev/null
        if [ ! -f "$local_file" ] || [ "$(wc -l < "$local_file.new")" -ge "$(wc -l < "$local_file")" ]; then
            mv "$local_file.new" "$local_file"
        else
            rm "$local_file.new"
        fi
        ssh -n $SSHOPTS -p "$3" "root@$2" "grep -c WRITE \$(dirname $remote)/progress.log || true" \
            | sed "s#^#  $1 round ${round:-1} done markers: #"
    done
done
wc -l runs/game_v2/full*/written_*.jsonl

echo "== check, vocabulary, corpus"
python -m library.domain.game_dsl.v2_corpus check | grep -v "^checked"
python -m library.domain.game_dsl.v2_corpus vocabulary
python -m library.domain.game_dsl.v2_corpus corpus

echo "== shares"
python - "$KIT" <<'EOF'
import json, sys
kit = sys.argv[1]
cards = [line.split()[0] for line in open(f"{kit}/cards.txt") if line.strip()]
rows = [line for line in open("library/domain/game_dsl/corpus_v2.jsonl", encoding="utf-8")]
train = [row for row in rows if '"split": "train"' in row]
test = [row for row in rows if '"split": "test"' in row]
assert len(train) + len(test) == len(rows)
for index, card in enumerate(cards):
    share = train[index::len(cards)] + test[index::len(cards)]       # each share: train first, test at its tail
    with open(f"{kit}/share_{card}.jsonl", "w", encoding="utf-8") as file:   # derived, rebuilt each launch
        file.writelines(share)
    print(card, len(share))
EOF

echo "== onto the cards"
vocab_md5=$(md5sum $GAME/output_tokens_v2.json | cut -d' ' -f1)
while read -r name ip port role; do
    scp -q $SSHOPTS -P "$port" "$KIT/share_$name.jsonl" "root@$ip:/workspace/pd/$GAME/corpus_v2.jsonl" < /dev/null
    scp -q $SSHOPTS -P "$port" "$GAME/output_tokens_v2.json" "root@$ip:/workspace/pd/$GAME/output_tokens_v2.json" < /dev/null
    remote=$(ssh -n $SSHOPTS -p "$port" "root@$ip" "md5sum /workspace/pd/$GAME/output_tokens_v2.json | cut -d' ' -f1; wc -l < /workspace/pd/$GAME/corpus_v2.jsonl")
    echo "  $name: vocabulary $( [ "$(echo "$remote" | head -1)" = "$vocab_md5" ] && echo same || echo DIFFERENT), $(echo "$remote" | tail -1) records"
    [ "$(echo "$remote" | head -1)" = "$vocab_md5" ] || { echo "vocabulary mismatch on $name: no GO"; exit 1; }
done < "$KIT/cards.txt"

echo "== the home data onto the game-v2 volume"
read -r keeper_ip keeper_port < "$KIT/keeper.txt"
tar -cf - runs/game_v2 $GAME/corpus_v2.jsonl $GAME/output_tokens_v2.json $GAME/problems/versions_round2.jsonl \
    $GAME/problems/versions_round3.jsonl \
    | ssh $SSHOPTS -p "$keeper_port" "root@$keeper_ip" "mkdir -p /vol/game_v2_home && tar -C /vol/game_v2_home -xf - && du -sh /vol/game_v2_home"

echo "== GO"
while read -r name ip port role; do
    ssh -n $SSHOPTS -p "$port" "root@$ip" "touch /workspace/GO && echo '  $name: GO'"
done < "$KIT/cards.txt"
date
