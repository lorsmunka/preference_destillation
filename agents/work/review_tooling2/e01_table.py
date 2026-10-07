import csv, statistics, json
rows = list(csv.DictReader(open(r"D:\preference_destillation\experiments\results\e01_anneal_direction.csv", encoding="utf-8")))
for dom in ["reddit", "math", "postgen"]:
    for arm in ["ce-to-kl", "kl-to-ce", "flat-50", "pure-ce", "pure-kl"]:
        g = [r for r in rows if r["domain"] == dom and r["arm"] == arm]
        out = []
        for k in ["task_accuracy_best", "task_accuracy_final", "teacher_student_overlap_final", "mean_target_rank_final"]:
            v = [float(r[k]) for r in g]
            out.append(f"{k[:14]}={statistics.mean(v):.4f} sd1={statistics.stdev(v):.4f} sd0={statistics.pstdev(v):.4f}")
        print(dom, arm, len(g), " | ".join(out))
# statuses
for r in rows:
    info = json.load(open(rf"D:\preference_destillation\runs\{r['run']}\info.json", encoding="utf-8"))
    if info.get("status") != "completed": print("NOT COMPLETED", r["run"], info.get("status"))
