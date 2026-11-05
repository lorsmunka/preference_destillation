"""The attention study's page: a local HTML file (never published), written from attention_report.compute().
One explorer (any reply token, any model, any layer) and six studies, each a question, its charts and
the numbers behind them. Charts are drawn in the page from the embedded numbers (plain SVG)."""

import base64
import json
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import attention_report as R
from attention_study import STUDY_DIR

PAGE_PATH = STUDY_DIR / "attention_study.html"
EXPLORER_MATH = {"solve": 3, "explain": 2, "route": 1}
EXPLORER_CODE = 2
EXPLORER_MODELS = {
    "math": ["gemma", "math-kl-m-s1", "math-ce-m-s1", "math-kl-to-ce-m-s1", "math-kl-m-s2", "math-size-xs", "math-size-l",
             "math-untrained-m"],
    "code": ["gemma", "code-kl-m", "code-size-xl", "code-size-s", "code-size-s-ce", "code-untrained-m"],
}
LOSS_METRICS = ["previous token", "copy", "numbers when writing a number", "distance", "entropy", "message",
                "newest task", "earlier conversation", "reply so far", "end of input"]


def rounded(values, digits=4):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def run_facts(students) -> Dict[str, dict]:
    from library import RunStore
    store = RunStore()
    facts = {}
    for student in students:
        entry = {"label": student.label, "domain": student.domain}
        if student.run is not None:
            run = store.get(student.run)
            final = run.eval_epochs[-1] if run.eval_epochs else None
            entry.update(params=round(run.params_millions(), 2), layers=run.info["num_layers"], heads=run.info["num_heads"],
                         passes=len(run.eval_epochs),
                         task_accuracy=None if final is None else final.task_accuracy,
                         baseline=None if final is None else run.majority_baseline(final))
        facts[student.key] = entry
    return facts


def encode_rows(view: torch.Tensor):
    """[layers + 1 (their mean), queries, places] -> (base64 of uint8 weight / row maximum, row maxima)."""
    maxima = view.amax(-1, keepdim=True).clamp(min=1e-9)
    scaled = (view / maxima * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    return base64.b64encode(scaled.tobytes()).decode(), rounded(maxima[..., 0], 5)


def explorer(results, probes) -> List[dict]:
    chosen = []
    remaining = dict(EXPLORER_MATH)
    for index, probe in enumerate(probes):
        if probe["kind"] == "math" and remaining.get(probe["bucket"], 0) > 0 and len(probe["student_queries"]) <= 60:
            remaining[probe["bucket"]] -= 1
            chosen.append(index)
    code_seen = set()
    for index, probe in enumerate(probes):
        if probe["kind"] == "code" and len(code_seen) < EXPLORER_CODE and probe["bucket"] not in code_seen \
                and (probe["bucket"] == "travel_site" or not code_seen):
            code_seen.add(probe["bucket"])
            chosen.append(index)
    entries = []
    for index in chosen:
        probe = probes[index]
        width = probe["group_count"] - 1
        entry = {"index": index, "kind": probe["kind"], "bucket": probe["bucket"], "item": probe["item_id"],
                 "labels": [label.replace("▁", " ") for label in probe["group_labels"][:width]],
                 "regions": probe["group_regions"][:width],
                 "reply": [token.replace("▁", " ") for token in probe["reply_tokens"]],
                 "queries": probe["student_queries"], "models": {}}
        if probe["kind"] == "code":
            entry["lines"] = [line[:90] for line in probe["sentence"].split("\n")]
        for key in EXPLORER_MODELS[probe["kind"]]:
            view, outside = R.view_of(R.load_rows(key)[index])
            stacked = torch.cat([view.mean(1), view.mean((0, 1))[None]])
            data, maxima = encode_rows(stacked)
            model = {"layers": view.shape[0], "data": data, "maxima": maxima}
            if key == R.GEMMA:
                model["outside"] = rounded(torch.cat([outside.mean(1), outside.mean((0, 1))[None]]), 3)
            entry["models"][key] = model
        entries.append(entry)
    return entries


def depth_alignment(math, key) -> List[List[float]]:
    """[student layer][Gemma layer]: mean over the student layer's heads of the best correlation with a
    head of that Gemma layer."""
    index, correlation = math["index"], math["correlation"]
    student_heads = R.heads_of(index, key)
    layers = sorted({index[position][1] for position in student_heads})
    gemma = R.heads_of(index, R.GEMMA)
    matrix = []
    for layer in layers:
        rows = [position for position in student_heads if index[position][1] == layer]
        matrix.append([float(correlation[np.ix_(rows, [g for g in gemma if index[g][1] == gemma_layer])].max(1).mean())
                       for gemma_layer in range(R.GEMMA_LAYERS)])
    return matrix


def matched_pairs(math, first, second):
    index, correlation = math["index"], math["correlation"]
    heads_first, heads_second = R.heads_of(index, first), R.heads_of(index, second)
    block = correlation[np.ix_(heads_first, heads_second)]
    rows, columns = linear_sum_assignment(-block)
    same_head = float(np.mean([index[heads_first[i]][1:] == index[heads_second[j]][1:] for i, j in zip(rows, columns)]))
    same_layer = float(np.mean([index[heads_first[i]][1] == index[heads_second[j]][1] for i, j in zip(rows, columns)]))
    return same_head, same_layer


def payload(results, probes) -> dict:
    math, code = results["math"], results["code"]
    facts = run_facts(math["students"] + code["students"])
    labels = {key: facts[key]["label"] for key in facts}
    labels[R.GEMMA] = "Gemma 4 12B (bf16)"
    seed = {}
    for name in ("overlap", "correlation"):
        study = math[f"seed_study_{name}"]
        matrix = [[1.0 if first == second else math[f"matched_{name}"][(first, second)] for second in R.SEED_MODELS]
                  for first in R.SEED_MODELS]
        seed[name] = {"matrix": rounded(matrix), "pairs": [[a, b, kind, round(value, 4)] for a, b, kind, value in study["pairs"]],
                      "summary": study["summary"], "p": study["p_value"], "permutations": study["permutations"],
                      "observed": study["observed"], "ratio": study["distance_ratio"]}
    whole = math["seed_study_whole_model"]
    seed["whole"] = {"summary": whole["summary"], "p": whole["p_value"], "observed": whole["observed"]}
    seed["identity"] = {f"{a}|{b}": matched_pairs(math, a, b) for a, b in
                        [("math-kl-m-s1", "math-ce-m-s1"), ("math-kl-m-s2", "math-ce-m-s2"), ("math-kl-m-s3", "math-ce-m-s3"),
                         ("math-kl-m-s1", "math-kl-m-s2"), ("math-kl-m-s1", "math-kl-m-s3"), ("math-kl-m-s2", "math-kl-m-s3"),
                         ("math-ce-m-s1", "math-ce-m-s2"), ("math-ce-m-s1", "math-ce-m-s3"), ("math-ce-m-s2", "math-ce-m-s3")]}
    untrained = {key: [R.matched_overlap(math["overlap"], math["index"], "math-untrained-m", key),
                       R.matched_overlap(math["correlation"], math["index"], "math-untrained-m", key)]
                 for key in R.SEED_MODELS}

    head_maps = {key: {metric: rounded(math["metrics"][key][metric], 4) for metric in LOSS_METRICS}
                 for key in R.SEED_MODELS + ["math-untrained-m"]}
    gemma_metrics = math["metrics"][R.GEMMA]
    gemma_profile = {metric: rounded(values.mean(1), 4) for metric, values in gemma_metrics.items()}
    gemma_maps = {metric: rounded(gemma_metrics[metric], 3) for metric in
                  ("outside view", "previous token", "self", "copy", "numbers when writing a number", "entropy", "message")}
    likeness = {key: {"mean": float(values.mean()), "heads": rounded(values, 3),
                      "overlap": float(math["gemma_overlap"][key].mean())}
                for key, values in math["gemma_correlation"].items()}
    patterns = {key: math["pattern_correlations"][key] for key in math["keys"]}

    passes = []
    for number in range(1, 11):
        kl = f"math-kl-m-s1-pass{number}" if number < 10 else "math-kl-m-s1"
        ce = f"math-ce-m-s1-pass{number}" if number < 10 else "math-ce-m-s1"
        passes.append({"pass": number,
                       "kl_final": R.matched_overlap(math["correlation"], math["index"], kl, "math-kl-m-s1"),
                       "ce_final": R.matched_overlap(math["correlation"], math["index"], ce, "math-ce-m-s1"),
                       "kl_ce": R.matched_overlap(math["correlation"], math["index"], kl, ce),
                       "kl_ce_overlap": R.matched_overlap(math["overlap"], math["index"], kl, ce),
                       "kl_gemma": likeness[kl]["mean"], "ce_gemma": likeness[ce]["mean"],
                       "kl_copy": float(math["metrics"][kl]["copy"].mean()), "ce_copy": float(math["metrics"][ce]["copy"].mean()),
                       "kl_entropy": float(math["metrics"][kl]["entropy"].mean()), "ce_entropy": float(math["metrics"][ce]["entropy"].mean()),
                       "kl_numbers": float(math["metrics"][kl]["numbers when writing a number"].mean()),
                       "ce_numbers": float(math["metrics"][ce]["numbers when writing a number"].mean())})

    def summary_metrics(result, key, names):
        return {name: float(result["metrics"][key][name].mean()) for name in names if name in result["metrics"][key]}

    region_names = R.MATH_REGIONS + [f"{region} [{bucket}]" for region in R.MATH_REGIONS for bucket in ("solve", "explain", "route")]
    regions = {key: summary_metrics(math, key, region_names + ["outside view"] + R.CATEGORIES)
               for key in math["keys"] if "pass" not in key}
    model_summary = {key: summary_metrics(math, key, LOSS_METRICS + ["self", "first token"]) for key in math["keys"] if "pass" not in key}
    for key in model_summary:
        model_summary[key]["best previous-token head"] = float(math["metrics"][key]["previous token"].max())
        model_summary[key]["best copy head"] = float(math["metrics"][key]["copy"].max())
        model_summary[key]["best number head"] = float(math["metrics"][key]["numbers when writing a number"].max())

    code_metrics = {}
    for key in code["keys"]:
        values = code["metrics"][key]
        code_metrics[key] = {name: {"mean": float(values[name].mean()), "best": float(values[name].max())}
                             for name in ("target lines", "target lift", "target lines when writing a line number", "message",
                                          "target file", "reply so far", "end of input", "outside view") if name in values}
    code_gemma_layers = {name: rounded(code["metrics"][R.GEMMA][name].max(1), 4)
                         for name in ("target lines", "target lines when writing a line number", "target lift")}
    code_gemma_layers_mean = {name: rounded(code["metrics"][R.GEMMA][name].mean(1), 4)
                              for name in ("target lines", "target lift", "message", "outside view")}

    return {
        "date": date.today().isoformat(),
        "labels": labels, "facts": facts,
        "probe_counts": {"math": sum(p["kind"] == "math" for p in probes), "code": sum(p["kind"] == "code" for p in probes),
                         "math_rows": sum(len(p["student_queries"]) for p in probes if p["kind"] == "math"),
                         "code_rows": sum(len(p["student_queries"]) for p in probes if p["kind"] == "code")},
        "seed_models": R.SEED_MODELS, "seed": seed, "untrained": untrained,
        "head_maps": head_maps, "loss_metrics": LOSS_METRICS,
        "gemma_profile": gemma_profile, "gemma_maps": gemma_maps, "gemma_full_layers": R.GEMMA_FULL_LAYERS,
        "likeness": likeness, "patterns": patterns,
        "depth": {key: depth_alignment(math, key) for key in ("math-kl-m-s1", "math-ce-m-s1", "math-kl-m-s2", "math-size-l", "math-size-xs")},
        "passes": passes, "regions": regions, "model_summary": model_summary,
        "code_metrics": code_metrics, "code_gemma_layers": code_gemma_layers, "code_gemma_layers_mean": code_gemma_layers_mean,
        "code_likeness": {key: float(values.mean()) for key, values in code["gemma_correlation"].items()},
        "explorer": explorer(results, probes),
    }


def write_page(results, probes) -> Path:
    data = payload(results, probes)
    template = (Path(__file__).resolve().parent / "attention_page_template.html").read_text(encoding="utf-8")
    page = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_PATH.write_text(page, encoding="utf-8")
    return PAGE_PATH
