"""Read-only: dump info.json key fields and every eval_epoch scalar of each run under runs/.

Writes two CSVs into this folder only: runs_info.csv, runs_evals.csv.
"""
import csv
import json
from pathlib import Path

ROOT = Path(r"D:\preference_destillation\runs")
OUT = Path(__file__).parent

info_fields = ["run_name", "domain", "experiment", "teacher_model", "hidden_dim", "num_layers", "num_heads",
               "epoch_count", "batch_size", "learning_rate", "max_training_examples", "max_test_examples",
               "shuffle_seed", "kl_ratio_start", "kl_ratio_end", "distillation_temperature_start",
               "distillation_temperature_end", "divergence", "status", "started_at", "completed_at",
               "training_batch_count", "test_batch_count", "longest_training_sequence", "optimizer_updates",
               "parameters", "description"]
info_rows = []
eval_rows = []
eval_keys = set()
for run_dir in sorted(ROOT.iterdir()):
    info_path = run_dir / "info.json"
    if not info_path.exists():
        continue
    info = json.loads(info_path.read_text(encoding="utf-8"))
    model_info = info.get("model_info") or {}
    parameters = None
    if isinstance(model_info, dict):
        for key in ("total_parameters", "parameters", "parameter_count", "total"):
            if key in model_info:
                parameters = model_info[key]
                break
    row = {field: info.get(field) for field in info_fields}
    row["parameters"] = parameters
    row["run_name"] = run_dir.name
    info_rows.append(row)
    log_path = run_dir / "logs" / "training.jsonl"
    if not log_path.exists():
        continue
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            if '"eval_epoch"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "eval_epoch":
                continue
            scalar = {"run": run_dir.name}
            for key, value in record.items():
                if isinstance(value, (int, float, str)) or value is None:
                    scalar[key] = value
                elif isinstance(value, dict) and key != "confusion_matrices":
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, str)):
                            scalar[f"{key}.{sub_key}"] = sub_value
            eval_keys.update(scalar.keys())
            eval_rows.append(scalar)

with open(OUT / "runs_info.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=info_fields)
    writer.writeheader()
    writer.writerows(info_rows)

ordered = ["run", "epoch", "timestamp"] + sorted(eval_keys - {"run", "epoch", "timestamp"})
with open(OUT / "runs_evals.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=ordered)
    writer.writeheader()
    writer.writerows(eval_rows)
print(len(info_rows), "runs;", len(eval_rows), "eval records;", len(ordered), "columns")
print(ordered)
