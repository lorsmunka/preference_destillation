import sys, csv, json, os
sys.path.insert(0, r"D:\preference_destillation"); sys.path.insert(0, r"D:\preference_destillation\experiments")
import anneal_direction as ad
rows = {r["run"]: r for r in csv.DictReader(open(r"D:\preference_destillation\experiments\results\e01_anneal_direction.csv", encoding="utf-8"))}
fresh = {r["run"]: r for r in ad.collect_rows(list(ad.DOMAINS))}
print("csv rows", len(rows), "fresh rows", len(fresh), "missing from csv", set(fresh) - set(rows), "extra in csv", set(rows) - set(fresh))
mismatch = 0
for name, f in fresh.items():
    c = rows.get(name)
    for k in ad.RESULT_COLUMNS:
        if c is None: continue
        a, b = str(f[k]), c[k]
        try:
            if abs(float(a) - float(b)) > 1e-9: mismatch += 1; print("diff", name, k, a, b)
        except ValueError:
            if a != b: mismatch += 1; print("diff", name, k, a, b)
print("value mismatches", mismatch)
# config check
spec = {run.run_name: run.as_dict() for _, run in ad.build_runs(list(ad.DOMAINS))}
bad = 0
for name, d in spec.items():
    info = json.load(open(rf"D:\preference_destillation\runs\{name}\info.json", encoding="utf-8"))
    for k, v in d.items():
        if k in ("description",): continue
        if info.get(k) != v:
            bad += 1
            if bad < 10: print("config diff", name, k, "spec", v, "info", info.get(k))
print("config diffs", bad)
