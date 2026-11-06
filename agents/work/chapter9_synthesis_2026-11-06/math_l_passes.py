"""CPU only, read-only: attention measures of the first-loop 53.7M student (math-full-kd-l-s1) after every pass,
on the attention study's 80 math probes (its test loss turns up after pass 8). Writes code_rows/math-l-pN.pt
and prints the head_metrics means. Same machinery as code_attention_passes.py."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, r"D:\preference_destillation\experiments")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import code_attention_passes as helper   # patches attention_report.load_rows to read code_rows/ first
import attention_report as report
import attention_study as study

torch.set_grad_enabled(False)
probes = helper.load_probes()
keys = []
for epoch in range(10):
    key = f"math-l-p{epoch + 1}"
    path = helper.ROWS_DIR / f"{key}.pt"
    if not path.exists():
        rows = study.student_rows(study.Student(key, "math-full-kd-l-s1", "math", key, epoch=epoch), probes, "cpu")
        torch.save(rows, path)
    keys.append(key)
names = ["entropy", "previous token", "copy", "numbers when writing a number", "distance", "message", "newest task",
         "earlier conversation", "reply so far"]
print("key".ljust(14), " | ".join(names))
for key in keys + ["math-size-m", "math-size-l"]:
    values = report.head_metrics(key, probes, "math")
    print(key.ljust(14), " | ".join(f"{float(values[name].mean()):.4f}" for name in names), flush=True)
