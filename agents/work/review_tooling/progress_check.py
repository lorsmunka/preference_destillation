import sys, json
sys.argv = ["progress.py"]
sys.path.insert(0, r"D:\preference_destillation")
import progress
infos = progress.run_infos()
print("current run:", progress.current_run(infos))
tags = {}
for n, (i, m) in infos.items(): tags.setdefault(i.get("experiment"), []).append(i)
for tag, lst in tags.items():
    try:
        shared, varying = progress.config_split(lst)
        print(tag, "varying:", {k: v for k, v in varying.items()})
    except Exception as e:
        print(tag, "CRASH", type(e).__name__, e)
s = progress.build_state()
print("experiment total", s["experiment"]["total"], "completed", s["experiment"]["completed"])
