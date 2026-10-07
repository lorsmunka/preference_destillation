import sys
sys.path.insert(0, r"D:\preference_destillation"); sys.path.insert(0, r"D:\preference_destillation\experiments")
import report_ideas as ri
from library import RunStore
completed = [r for r in RunStore().all() if r.status == "completed"]
for prefix, key, label in [("ideaA-", lambda r: r.output_vocab_size, "V"), ("ideaD-", lambda r: r.info.get("max_training_examples", 0), "examples")]:
    rows = ri.cohort_rows([r for r in completed if r.name.startswith(prefix)], key, label)
    for row in rows:
        print(prefix, label, row[label], "runs", row["runs"], "task_best", round(row["task_best"], 4), "+/-", round(row["task_best_std"], 4), "task_final", round(row["task_final"], 4), "teacher_H", round(row["teacher_entropy"], 5))
