import sys
sys.path.insert(0, r"D:\preference_destillation")
from library import Analysis, RunStore
from library.tooling.render import experiment_summary, comparison_table
cohorts = Analysis.cohorts()
for tag, exp in cohorts.items():
    statuses = {}
    for r in exp.runs: statuses[r.status] = statuses.get(r.status, 0) + 1
    print(tag, len(exp), statuses, "domains:", sorted({r.domain for r in exp.runs}))
print(experiment_summary(list(cohorts.values())))
store = RunStore()
print(comparison_table(store.completed()[:3]))
# training hours vs session count
import json
for r in store.all():
    sessions = set()
    for line in open(r.run_dir / "logs" / "training.jsonl", encoding="utf-8"):
        rec = json.loads(line); sessions.add(rec.get("session_id"))
    if len(sessions) > 1:
        tb = sum(b.time_seconds for b in r.train_batches) / 3600
        print("multi-session:", r.name, r.status, len(sessions), "training_hours(info)=", None if r.training_hours is None else round(r.training_hours, 3), "sum batch time h=", round(tb, 3))
