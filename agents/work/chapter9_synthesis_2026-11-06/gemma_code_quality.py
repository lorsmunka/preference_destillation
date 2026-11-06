"""Read-only: Gemma's own scope quality on the coding agent's v1 pass (conversations/v1.jsonl), by split."""
import json
import statistics
from collections import defaultdict

path = r"D:\preference_destillation\library\domain\code_agent\conversations\v1.jsonl"
groups = defaultdict(list)
for line in open(path, encoding="utf-8"):
    record = json.loads(line)
    turn = record["turn"]
    groups[record["split"]].append(turn)
for split, turns in groups.items():
    recall = [t["recall"] for t in turns if t.get("recall") is not None]
    precision = [t["precision"] for t in turns if t.get("precision") is not None]
    hit = sum(t.get("places_hit") or 0 for t in turns)
    places = sum(t.get("places") or 0 for t in turns)
    route = sum(1 for t in turns if t.get("route_correct"))
    exact = sum(1 for t in turns if t.get("recall") == 1.0 and t.get("precision") == 1.0)
    first_try = sum(1 for t in turns if t.get("scope_attempts") == 1)
    complete = sum(1 for t in turns)
    kinds = defaultdict(int)
    for t in turns:
        kinds[t.get("kind")] += 1
    print(f"{split}: {len(turns)} requests, kinds {dict(kinds)}; route right {route}/{len(turns)} ({route/len(turns):.3f}); "
          f"line recall mean {statistics.mean(recall):.3f}, precision mean {statistics.mean(precision):.3f}; "
          f"places hit {hit}/{places} ({hit/places:.3f}); scope = true places exactly {exact}/{len(turns)}; scope on first try {first_try}")
