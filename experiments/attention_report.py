"""The attention study's numbers (experiments/attention_study.py, report step): per-head metrics, the
overlap of every head with every other head, and the comparisons built on it. The page is written by
attention_page.py from what this returns.

The overlap of two attention rows is the weight they put on the same places: sum over places of
min(a, b), from 0 (disjoint) to 1 (identical); it equals 1 minus their total variation distance.
Two heads' overlap is that, averaged over every reply position of every probe. Gemma's rows are read
without the weight it puts outside the student's view (renormalised), so both spread 1 over the same
places.

A model-level number needs heads matched, since a head's index means nothing across trainings:
  matched overlap   one-to-one matching of the two students' heads maximising the total (Hungarian),
                    then the mean over the pairs (same head count; seeds and losses at size m)
  Gemma likeness    for each student head, its best-matching head among Gemma's 768; the mean
"""

import itertools
import random
import re
import statistics
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from attention_study import (STUDENTS_DIR, TEACHER_PATHS, Student, code_students, math_students, scope_target_lines,
                             sentiment_students)

GEMMA = "gemma"
GEMMA_LAYERS, GEMMA_HEADS = 48, 16
GEMMA_FULL_LAYERS = [layer for layer in range(GEMMA_LAYERS) if layer % 6 == 5]   # every 6th layer sees everything
PATTERNS = ("previous token", "uniform", "first token")
MATH_REGIONS = ["message", "newest task", "earlier conversation", "end of input", "reply so far"]
CODE_REGIONS = ["target lines", "target file", "other files", "message", "markup", "end of input", "reply so far"]
CATEGORIES = ["number", "word", "markup", "space"]
SEED_MODELS = ["math-kl-m-s1", "math-kl-m-s2", "math-kl-m-s3", "math-ce-m-s1", "math-ce-m-s2", "math-ce-m-s3",
               "math-kl-to-ce-m-s1", "math-kl-to-ce-m-s2"]


SENTIMENT_REGIONS = ["comment", "end of input", "reply so far"]
SENTIMENT_ARM_MODELS = [f"sent-{arm}-s{seed}" for arm in ("pure-kl", "pure-ce", "kl-to-ce", "ce-to-kl", "flat-50") for seed in range(1, 11)]
TOKEN_KINDS = ("math", "sentiment")   # kinds whose groups are the student's tokens (code: the project's lines)


def regions_of(kind: str) -> List[str]:
    return {"math": MATH_REGIONS, "code": CODE_REGIONS, "sentiment": SENTIMENT_REGIONS}[kind]


def setting_of(key: str) -> str:
    """The run's settings without its seed: math-kl-m-s2 -> math-kl-m, sent-pure-ce-s7 -> sent-pure-ce."""
    return key.rsplit("-s", 1)[0]


def seed_of(key: str) -> int:
    return int(key.rsplit("-s", 1)[1])


def load_rows(key: str) -> Dict[int, torch.Tensor]:
    if key == GEMMA:   # each kind's own teacher (Gemma 4 12B for the agents, Gemma 3 4B for sentiment)
        merged = {}
        for path in TEACHER_PATHS.values():
            if path.exists():
                merged.update(torch.load(path, mmap=True, weights_only=False))
        return merged
    return torch.load(STUDENTS_DIR / f"{key}.pt", mmap=True, weights_only=False)


def view_of(rows: torch.Tensor):
    """(rows over the student's places, renormalised; weight outside the student's view) from
    [layers, heads, queries, groups] whose last group is "outside"."""
    rows = rows.float()
    outside = rows[..., -1]
    inside = rows[..., :-1]
    return inside / inside.sum(-1, keepdim=True).clamp(min=1e-9), outside


def pattern_rows(probe: dict) -> torch.Tensor:
    """[1, 3, queries, groups - 1]: where a head that only looked at the previous token, at everything
    equally, or at the first token would put its weight, in the probe's groups."""
    groups = torch.tensor(probe["student_groups"])
    width = probe["group_count"] - 1
    rows = torch.zeros(3, len(probe["student_queries"]), width)
    for row, query in enumerate(probe["student_queries"]):
        rows[0, row, groups[query - 1]] = 1.0
        rows[1, row].index_add_(0, groups[:query + 1], torch.full((query + 1,), 1.0 / (query + 1)))
        rows[2, row, groups[0]] = 1.0
    return rows[None]


# ── per-head metrics ─────────────────────────────────────────────────────────────────────────────────

def writing_digits(probe: dict) -> List[int]:
    """The query rows that write a digit (a number in a solution, a line number in a scope)."""
    return [row for row, token in enumerate(probe["reply_tokens"]) if token.replace("▁", "").isdigit()]


def probe_masks(probe: dict):
    """Per query row: masks over the groups for each metric (math: groups = the student's tokens)."""
    width = probe["group_count"] - 1
    queries = probe["student_queries"]
    regions = probe["group_regions"][:width]
    masks = {}
    for name in regions_of(probe["kind"]):
        masks[name] = torch.tensor([region == name for region in regions], dtype=torch.float32)
    if probe["kind"] in TOKEN_KINDS:
        categories = probe["group_categories"][:width]
        for name in CATEGORIES:
            masks[name] = torch.tensor([category == name for category in categories], dtype=torch.float32)
    return masks


def head_metrics(key: str, probes: List[dict], kind: str) -> Dict[str, np.ndarray]:
    """Mean over every reply position of the probes of one kind, per layer and head:
    math: previous token, self, first token, look-back distance (tokens), entropy (share of its maximum),
    copy (weight on earlier copies of the token about to be written, rows where one exists), the regions,
    the token categories, and "numbers when writing a number"; code: the regions and the lift of the
    target lines (their weight over their share of the project's lines). Gemma also: outside its view."""
    rows_by_probe = load_rows(key)
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, float] = {}

    def add(name, values, count):
        sums[name] = sums.get(name, 0) + values
        counts[name] = counts.get(name, 0) + count

    for index, probe in enumerate(probes):
        if probe["kind"] != kind:
            continue
        weights, outside = view_of(rows_by_probe[index])
        rows = weights.shape[2]
        add("outside view", outside.sum(2), rows)
        masks = probe_masks(probe)
        for name, mask in masks.items():
            add(name, (weights * mask).sum((2, 3)), rows)
            if kind == "math" and name in MATH_REGIONS:
                add(f"{name} [{probe['bucket']}]", (weights * mask).sum((2, 3)), rows)
        targets = probe["target_token_ids"]
        if kind in TOKEN_KINDS:
            queries = torch.tensor(probe["student_queries"])
            positions = torch.arange(weights.shape[3])
            rows_index = torch.arange(rows)
            add("previous token", weights[:, :, rows_index, queries - 1].sum(2), rows)
            add("self", weights[:, :, rows_index, queries].sum(2), rows)
            add("first token", weights[..., 0].sum(2), rows)
            distance = (queries[:, None] - positions[None, :]).clamp(min=0).float()
            add("distance", (weights * distance).sum((2, 3)), rows)
            entropy = -(weights * weights.clamp(min=1e-12).log()).sum(-1) / (queries.float() + 1).log()[None, None]
            add("entropy", entropy.sum(2), rows)
            token_ids = torch.tensor(probe["group_token_ids"][:weights.shape[3]])
            copy_rows = [row for row, target in enumerate(targets) if (token_ids[:queries[row] + 1] == target).any()]
            if copy_rows:
                copy_mask = torch.stack([(token_ids == targets[row]).float() * (positions <= queries[row]).float()
                                         for row in copy_rows])
                add("copy", (weights[:, :, copy_rows] * copy_mask).sum((2, 3)), len(copy_rows))
            digit_rows = writing_digits(probe)
            if digit_rows:
                add("numbers when writing a number", (weights[:, :, digit_rows] * masks["number"]).sum((2, 3)), len(digit_rows))
        else:
            # what a head spreading evenly over the tokens would give the target lines (they differ in length)
            uniform_share = float((pattern_rows(probe)[0, 1] * masks["target lines"]).sum())
            if uniform_share > 0:
                add("target mass", (weights * masks["target lines"]).sum((2, 3)), rows)
                add("uniform target mass", torch.full(weights.shape[:2], uniform_share), rows)
            line_number_rows = writing_digits(probe)
            if line_number_rows:
                add("target lines when writing a line number", (weights[:, :, line_number_rows] * masks["target lines"]).sum((2, 3)),
                    len(line_number_rows))
    metrics = {name: (sums[name] / counts[name]).numpy() for name in sums}
    if "target mass" in metrics:   # the lift of the target lines: how many times the even spread's weight
        metrics["target lift"] = metrics.pop("target mass") / metrics.pop("uniform target mass")
    return metrics


# ── overlaps ─────────────────────────────────────────────────────────────────────────────────────────

def head_index(keys: List[str], probes: List[dict], kind: str):
    """[(key, layer, head)] for every head of every model, Gemma's included, then the patterns."""
    first = next(index for index, probe in enumerate(probes) if probe["kind"] == kind)
    index = []
    for key in keys:
        layers, heads = load_rows(key)[first].shape[:2]
        index += [(key, layer, head) for layer in range(layers) for head in range(heads)]
    index += [("pattern", 0, number) for number in range(len(PATTERNS))]
    return index


def overlap_matrix(keys: List[str], probes: List[dict], kind: str, device: str) -> np.ndarray:
    """Head-by-head overlap over every reply position of every probe of one kind."""
    loaded = {key: load_rows(key) for key in keys}
    total, row_count = None, 0
    for index, probe in enumerate(probes):
        if probe["kind"] != kind:
            continue
        parts = [view_of(loaded[key][index])[0] for key in keys] + [pattern_rows(probe)]
        flat = torch.cat([part.reshape(-1, part.shape[2] * part.shape[3]) for part in parts]).to(device)
        rows = parts[0].shape[2]
        distance = torch.cdist(flat[None], flat[None], p=1)[0]
        contribution = rows - distance / 2
        total = contribution if total is None else total + contribution
        row_count += rows
        del flat, distance
    return (total / row_count).cpu().numpy()


def valid_places(probe: dict) -> torch.Tensor:
    """[queries, groups - 1]: the places a reply position can see (the groups of positions 0..query)."""
    groups = torch.tensor(probe["student_groups"])
    valid = torch.zeros(len(probe["student_queries"]), probe["group_count"] - 1, dtype=torch.bool)
    for row, query in enumerate(probe["student_queries"]):
        valid[row, groups[:query + 1]] = True
    return valid


def standardised(rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Each row centred and scaled to length 1 over the places it can see (0 elsewhere); a flat row
    (uniform attention) becomes all zeros, so it correlates with nothing."""
    mask = valid.float()[None, None]
    count = mask.sum(-1, keepdim=True)
    centred = (rows - (rows * mask).sum(-1, keepdim=True) / count) * mask
    length = centred.norm(dim=-1, keepdim=True)
    return torch.where(length > 1e-8, centred / length.clamp(min=1e-12), torch.zeros_like(centred))


def correlation_matrix(keys: List[str], probes: List[dict], kind: str, device: str) -> np.ndarray:
    """Head-by-head Pearson correlation of attention rows (over the places each row can see), averaged
    over every reply position: high when two heads favour the same tokens, whatever their breadth."""
    loaded = {key: load_rows(key) for key in keys}
    total, row_count = None, 0
    for index, probe in enumerate(probes):
        if probe["kind"] != kind:
            continue
        valid = valid_places(probe)
        parts = [standardised(view_of(loaded[key][index])[0], valid) for key in keys]
        parts.append(standardised(pattern_rows(probe), valid))
        flat = torch.cat([part.reshape(-1, part.shape[2] * part.shape[3]) for part in parts]).to(device)
        contribution = flat @ flat.T
        total = contribution if total is None else total + contribution
        row_count += parts[0].shape[2]
    return (total / row_count).cpu().numpy()


def model_mean_overlap(keys: List[str], probes: List[dict], kind: str, device: str):
    """Overlap of whole-model attention (every head and layer averaged), and Gemma's per-layer means,
    across models: needs no head matching, so any two models compare, whatever their sizes."""
    loaded = {key: load_rows(key) for key in keys}
    first = next(index for index, probe in enumerate(probes) if probe["kind"] == kind)
    gemma_layers = loaded[GEMMA][first].shape[0]
    names = list(keys) + [f"gemma layer {layer + 1}" for layer in range(gemma_layers)] + list(PATTERNS)
    total, row_count = None, 0
    for index, probe in enumerate(probes):
        if probe["kind"] != kind:
            continue
        means = [view_of(loaded[key][index])[0].mean((0, 1)) for key in keys]
        gemma_view = view_of(loaded[GEMMA][index])[0]
        means += [gemma_view[layer].mean(0) for layer in range(gemma_layers)]
        means += list(pattern_rows(probe)[0])
        flat = torch.stack([mean.reshape(-1) for mean in means]).to(device)
        rows = means[0].shape[0]
        contribution = rows - torch.cdist(flat[None], flat[None], p=1)[0] / 2
        total = contribution if total is None else total + contribution
        row_count += rows
    return names, (total / row_count).cpu().numpy()


def heads_of(index, key):
    return [position for position, (owner, _, _) in enumerate(index) if owner == key]


def matched_overlap(overlap: np.ndarray, index, first: str, second: str) -> float:
    """Mean overlap of the one-to-one head matching that maximises the total (Hungarian)."""
    block = overlap[np.ix_(heads_of(index, first), heads_of(index, second))]
    rows, columns = linear_sum_assignment(-block)
    return float(block[rows, columns].mean())


def best_match(overlap: np.ndarray, index, key: str, among: str) -> np.ndarray:
    """For each head of `key`, its overlap with the best-matching head of `among`."""
    return overlap[np.ix_(heads_of(index, key), heads_of(index, among))].max(1)


def pattern_overlaps(overlap: np.ndarray, index, key: str) -> Dict[str, float]:
    patterns = heads_of(index, "pattern")
    block = overlap[np.ix_(heads_of(index, key), patterns)]
    return {name: float(block[:, number].mean()) for number, name in enumerate(PATTERNS)}


# ── the seed baseline and the loss comparison ────────────────────────────────────────────────────────

def pair_kind(first: str, second: str) -> str:
    same_setting, same_seed = setting_of(first) == setting_of(second), seed_of(first) == seed_of(second)
    if same_setting:
        return "same loss, other seed"
    return "other loss, same seed" if same_seed else "other loss, other seed"


def seed_study(similarity: Dict[tuple, float], models: List[str] = None, samples: int = 20000) -> dict:
    """Pairs of the size-m students by kind, and a permutation test: is a pair with the same loss more
    alike than a pair with another loss (both from different seeds)? The losses are shuffled within each
    seed (math: seeds 1 and 2 have KL, CE and KL to CE, seed 3 KL and CE: 3! * 3! * 2! = 72 relabellings, all
    of them; sentiment: 5 schedules x 10 seeds, (5!)^10 relabellings, `samples` of them drawn at random)."""
    models = models or SEED_MODELS
    pairs = [(first, second, pair_kind(first, second), similarity[(first, second)])
             for first, second in itertools.combinations(models, 2)]
    by_kind = {}
    for _, _, kind, value in pairs:
        by_kind.setdefault(kind, []).append(value)

    def statistic(settings):
        same, other = [], []
        for first, second in itertools.combinations(models, 2):
            if seed_of(first) == seed_of(second):
                continue
            (same if settings[first] == settings[second] else other).append(similarity[(first, second)])
        return statistics.mean(same) - statistics.mean(other)

    observed = statistic({key: setting_of(key) for key in models})
    blocks = {}
    for key in models:
        blocks.setdefault(seed_of(key), []).append(key)
    total = 1
    for block in blocks.values():
        total *= len(list(itertools.permutations(block)))
    if total <= samples:
        orderings = itertools.product(*[list(itertools.permutations(block)) for block in blocks.values()])
    else:
        generator = random.Random(0)
        orderings = ([generator.sample(block, len(block)) for block in blocks.values()] for _ in range(samples))
    statistics_seen = []
    for orders in orderings:
        relabelled = {}
        for block, order in zip(blocks.values(), orders):
            for original, shuffled in zip(block, order):
                relabelled[original] = setting_of(shuffled)
        statistics_seen.append(statistic(relabelled))
    p_value = sum(value >= observed - 1e-12 for value in statistics_seen) / len(statistics_seen)
    summary = {kind: {"mean": statistics.mean(values), "min": min(values), "max": max(values), "n": len(values)}
               for kind, values in by_kind.items()}
    floor = 1 - summary["same loss, other seed"]["mean"]
    return {"pairs": pairs, "summary": summary, "observed": observed, "p_value": p_value,
            "permutations": len(statistics_seen),
            "distance_ratio": {kind: (1 - values["mean"]) / floor for kind, values in summary.items()}}


def compute(probes: List[dict]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for kind, students in (("math", math_students()), ("code", code_students()), ("sentiment", sentiment_students())):
        keys = [student.key for student in students] + [GEMMA]
        print(f"{kind}: overlaps of {len(keys)} models ...")
        index = head_index(keys, probes, kind)
        overlap = overlap_matrix(keys, probes, kind, device)
        names, mean_overlap = model_mean_overlap(keys, probes, kind, device)
        metrics = {key: head_metrics(key, probes, kind) for key in keys}
        correlation = correlation_matrix(keys, probes, kind, device)
        results[kind] = {"students": students, "keys": keys, "index": index, "overlap": overlap,
                         "correlation": correlation,
                         "mean_names": names, "mean_overlap": mean_overlap, "metrics": metrics,
                         "gemma_overlap": {s.key: best_match(overlap, index, s.key, GEMMA) for s in students},
                         "gemma_correlation": {s.key: best_match(correlation, index, s.key, GEMMA) for s in students},
                         "patterns": {key: pattern_overlaps(overlap, index, key) for key in keys},
                         "pattern_correlations": {key: pattern_overlaps(correlation, index, key) for key in keys}}
    math = results["math"]
    for name, matrix in (("overlap", math["overlap"]), ("correlation", math["correlation"])):
        similarity = {}
        for first, second in itertools.combinations(SEED_MODELS, 2):
            similarity[(first, second)] = similarity[(second, first)] = matched_overlap(matrix, math["index"], first, second)
        math[f"matched_{name}"] = similarity
        math[f"seed_study_{name}"] = seed_study(similarity)
    names = math["mean_names"]
    mean_similarity = {(first, second): float(math["mean_overlap"][names.index(first), names.index(second)])
                       for first in SEED_MODELS for second in SEED_MODELS if first != second}
    math["seed_study_whole_model"] = seed_study(mean_similarity)
    math["bootstrap"] = bootstrap_seed_study(probes, device)
    math["content"] = content_likeness([student.key for student in math["students"]], probes, device)
    results["code"]["retrieval_heads"] = retrieval_heads(probes, results["code"])
    sentiment = results["sentiment"]
    for name, matrix in (("overlap", sentiment["overlap"]), ("correlation", sentiment["correlation"])):
        similarity = {}
        for first, second in itertools.combinations(SENTIMENT_ARM_MODELS, 2):
            similarity[(first, second)] = similarity[(second, first)] = matched_overlap(matrix, sentiment["index"], first, second)
        sentiment[f"matched_{name}"] = similarity
        sentiment[f"seed_study_{name}"] = seed_study(similarity, SENTIMENT_ARM_MODELS)
    sentiment["data_models"] = [s.key for s in sentiment["students"] if s.key.startswith("sent-data")]
    data_similarity = {}
    for first, second in itertools.combinations(sentiment["data_models"], 2):
        data_similarity[(first, second)] = data_similarity[(second, first)] = matched_overlap(
            sentiment["correlation"], sentiment["index"], first, second)
    sentiment["matched_data"] = data_similarity
    sentiment["content"] = content_likeness([s.key for s in sentiment["students"]], probes, device, kind="sentiment")
    return results


def content_likeness(keys: List[str], probes: List[dict], device: str, kind: str = "math") -> Dict[str, dict]:
    """Gemma likeness with position taken out (the adversarial check, 2026-11-05): much of any two
    heads' likeness is shared recency. Each head's position-only version gives every visible place the
    head's mean weight at that look-back distance (over every reply position of the math probes); the
    residual (real minus position-only) keeps only what depends on the tokens themselves. Returns per
    student: the best-match correlation with Gemma's heads on real rows and on residual rows, overall and
    for Gemma's full-attention vs sliding layers (mean over those layers of the best head in each)."""
    math_probes = [index for index, probe in enumerate(probes) if probe["kind"] == kind]
    loaded = {key: load_rows(key) for key in keys + [GEMMA]}
    gemma_layers, gemma_heads = loaded[GEMMA][math_probes[0]].shape[:2]
    longest = max(max(probes[index]["student_queries"]) for index in math_probes) + 1

    def heads_view(key, index):
        view = view_of(loaded[key][index])[0]
        return view.reshape(-1, view.shape[2], view.shape[3]).to(device)

    def distances(probe):
        queries = torch.tensor(probe["student_queries"], device=device)
        distance = queries[:, None] - torch.arange(probe["group_count"] - 1, device=device)[None, :]
        return distance, distance >= 0

    sums, counts = {}, torch.zeros(longest, device=device)
    for index in math_probes:
        distance, visible = distances(probes[index])
        flat_distance = distance[visible]
        counts.index_add_(0, flat_distance, torch.ones_like(flat_distance, dtype=torch.float32))
        for key in keys + [GEMMA]:
            weights = heads_view(key, index)[:, visible]
            sums.setdefault(key, torch.zeros(weights.shape[0], longest, device=device)).index_add_(1, flat_distance, weights)
    profiles = {key: total / counts.clamp(min=1)[None] for key, total in sums.items()}

    def standardise(rows, visible):
        mask = visible.float()[None]
        centred = (rows - (rows * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)) * mask
        length = centred.norm(dim=-1, keepdim=True)
        return torch.where(length > 1e-8, centred / length.clamp(min=1e-12), torch.zeros_like(centred))

    totals = {key: {"real": 0, "residual": 0} for key in keys}
    row_count = 0
    for index in math_probes:
        distance, visible = distances(probes[index])

        def both(key):
            real = heads_view(key, index)
            position_only = profiles[key][:, distance.clamp(min=0)] * visible.float()[None]
            position_only = position_only / position_only.sum(-1, keepdim=True).clamp(min=1e-12)
            return (standardise(real, visible).reshape(real.shape[0], -1),
                    standardise(real - position_only, visible).reshape(real.shape[0], -1))
        gemma_real, gemma_residual = both(GEMMA)
        for key in keys:
            real, residual = both(key)
            totals[key]["real"] = totals[key]["real"] + real @ gemma_real.T
            totals[key]["residual"] = totals[key]["residual"] + residual @ gemma_residual.T
        row_count += len(probes[index]["student_queries"])
    full = [layer for layer in range(gemma_layers) if layer % 6 == 5]   # both Gemmas: every 6th layer is global
    sliding = [layer for layer in range(gemma_layers) if layer not in full]

    def best(matrix, layers=None):
        if layers is None:
            return float(matrix.max(1).mean())
        return float(np.mean([matrix[:, layer * gemma_heads:(layer + 1) * gemma_heads].max(1).mean() for layer in layers]))

    result = {}
    for key in keys:
        entry = {}
        for name in ("real", "residual"):
            matrix = (totals[key][name] / row_count).cpu().numpy()
            entry.update({name: best(matrix), f"{name} full": best(matrix, full), f"{name} sliding": best(matrix, sliding)})
        result[key] = entry
    return result


def bootstrap_seed_study(probes: List[dict], device: str, resamples: int = 1000) -> dict:
    """95% intervals over the probes (resampled with replacement, the head matching redone each time):
    the three pair kinds, same loss minus other loss (both across seeds), and KL minus CE Gemma likeness
    per seed. The probes are the sample here; the permutation test treats the students as the sample."""
    keys = SEED_MODELS + [GEMMA]
    loaded = {key: load_rows(key) for key in keys}
    contributions, row_counts = [], []
    for index, probe in enumerate(probes):
        if probe["kind"] != "math":
            continue
        valid = valid_places(probe)
        parts = [standardised(view_of(loaded[key][index])[0], valid) for key in keys]
        flat = torch.cat([part.reshape(-1, part.shape[2] * part.shape[3]) for part in parts]).to(device)
        students = sum(part.shape[0] * part.shape[1] for part in parts[:-1])
        contributions.append((flat[:students] @ flat.T).cpu().numpy())
        row_counts.append(parts[0].shape[2])
    contributions, row_counts = np.stack(contributions), np.array(row_counts)
    heads = 24
    offsets = {key: number * heads for number, key in enumerate(SEED_MODELS)}
    gemma_start = len(SEED_MODELS) * heads
    generator = np.random.default_rng(0)
    samples = {"same loss, other seed": [], "other loss, same seed": [], "other loss, other seed": [], "same minus other loss": [],
               "kl minus ce likeness, seed 1": [], "kl minus ce likeness, seed 2": [], "kl minus ce likeness, seed 3": [],
               "kl minus ce likeness, mean": []}
    for _ in range(resamples):
        chosen = generator.integers(0, len(row_counts), len(row_counts))
        matrix = contributions[chosen].sum(0) / row_counts[chosen].sum()
        by_kind = {}
        for first, second in itertools.combinations(SEED_MODELS, 2):
            block = matrix[offsets[first]:offsets[first] + heads, offsets[second]:offsets[second] + heads]
            rows, columns = linear_sum_assignment(-block)
            by_kind.setdefault(pair_kind(first, second), []).append(block[rows, columns].mean())
        for kind, values in by_kind.items():
            samples[kind].append(np.mean(values))
        samples["same minus other loss"].append(np.mean(by_kind["same loss, other seed"]) - np.mean(by_kind["other loss, other seed"]))
        likeness = {key: matrix[offsets[key]:offsets[key] + heads, gemma_start:].max(1).mean() for key in SEED_MODELS}
        differences = [likeness[f"math-kl-m-s{seed}"] - likeness[f"math-ce-m-s{seed}"] for seed in (1, 2, 3)]
        for seed, difference in zip((1, 2, 3), differences):
            samples[f"kl minus ce likeness, seed {seed}"].append(difference)
        samples["kl minus ce likeness, mean"].append(np.mean(differences))
    return {name: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)),
                   "median": float(np.median(values))} for name, values in samples.items()}


def retrieval_heads(probes: List[dict], code: dict, count: int = 10) -> List[dict]:
    """Gemma's heads with the most weight on the target lines while writing line numbers, the best head of
    two students for comparison, and in how many probes each head's most-attended line (its weight over
    the line-number rows, averaged) lies inside the target lines."""
    metric = "target lines when writing a line number"
    gemma_values = code["metrics"][GEMMA][metric]
    order = np.argsort(gemma_values.ravel())[::-1][:count]
    chosen = [(GEMMA, *np.unravel_index(position, gemma_values.shape)) for position in order]
    for key in ("code-kl-m", "code-size-xl"):
        values = code["metrics"][key][metric]
        chosen.append((key, *np.unravel_index(values.argmax(), values.shape)))
    loaded = {key: load_rows(key) for key in {key for key, _, _ in chosen}}
    hits = {head: 0 for head in chosen}
    file_hits = {head: 0 for head in chosen}
    probe_count = 0
    for index, probe in enumerate(probes):
        if probe["kind"] != "code":
            continue
        rows = writing_digits(probe)
        lines = probe["sentence"].split("\n")
        target = [region == "target lines" for region in probe["group_regions"][:len(lines)]]
        if not rows or not any(target):
            continue
        probe_count += 1
        # while writing the file name (after the "<scope>" line, before the first digit): which file is looked at?
        file_of_line, current_file = [], None
        for line in lines:
            match = re.match(r'<file name="([^"]+)">', line)
            current_file = match.group(1) if match else current_file
            file_of_line.append(current_file)
        target_files = set(scope_target_lines(probe["reply"]))
        newline = next((row for row, token in enumerate(probe["reply_tokens"]) if "\n" in token or "Ċ" in token), None)
        name_rows = list(range(newline + 1, rows[0])) if newline is not None and newline + 1 < rows[0] else []
        for head in chosen:
            key, layer, number = head
            head_rows = view_of(loaded[key][index])[0][layer, number]
            hits[head] += bool(target[int(head_rows[rows, :len(lines)].mean(0).argmax())])
            if name_rows:
                file_hits[head] += file_of_line[int(head_rows[name_rows, :len(lines)].mean(0).argmax())] in target_files
    return [{"model": key, "layer": int(layer) + 1, "head": int(number) + 1,
             "share": float(code["metrics"][key][metric][layer, number]), "hits": hits[(key, layer, number)],
             "file_hits": file_hits[(key, layer, number)], "probes": probe_count}
            for key, layer, number in chosen]


# ── the report step ──────────────────────────────────────────────────────────────────────────────────

MODEL_COLUMNS = ["model", "domain", "label", "gemma_likeness_correlation", "gemma_likeness_overlap", "entropy",
                 "previous_token", "copy", "numbers_when_writing_a_number", "distance_tokens", "message", "reply_so_far",
                 "outside_view", "target_lift", "target_lines_when_writing_a_line_number"]
PAIR_COLUMNS = ["pair", "kind", "matched_correlation", "matched_overlap"]
TEACHER_LABELS = {"math": "Gemma 4 12B (bf16)", "code": "Gemma 4 12B (bf16)", "sentiment": "Gemma 3 4B (bf16)"}


def summary_rows(results) -> List[dict]:
    rows = []
    for kind, result in results.items():
        labels = {student.key: student.label for student in result["students"]}
        labels[GEMMA] = TEACHER_LABELS[kind]
        for key in result["keys"]:
            metrics = result["metrics"][key]

            def mean(name):
                return round(float(metrics[name].mean()), 4) if name in metrics else ""
            # the teacher has one row per kind (one shared "gemma" key let the last kind overwrite the others)
            rows.append({"model": f"{GEMMA}-{kind}" if key == GEMMA else key, "domain": kind, "label": labels[key],
                         "gemma_likeness_correlation": round(float(result["gemma_correlation"][key].mean()), 4) if key != GEMMA else "",
                         "gemma_likeness_overlap": round(float(result["gemma_overlap"][key].mean()), 4) if key != GEMMA else "",
                         "entropy": mean("entropy"), "previous_token": mean("previous token"), "copy": mean("copy"),
                         "numbers_when_writing_a_number": mean("numbers when writing a number"), "distance_tokens": mean("distance"),
                         "message": mean("message"), "reply_so_far": mean("reply so far"), "outside_view": mean("outside view"),
                         "target_lift": mean("target lift"),
                         "target_lines_when_writing_a_line_number": mean("target lines when writing a line number")})
    return rows


def report(probes: List[dict]) -> None:
    from _common import write_rows
    from attention_page import write_page
    results = compute(probes)
    write_rows("attention_models.csv", MODEL_COLUMNS, summary_rows(results))
    math = results["math"]
    pairs = [{"pair": f"{first} | {second}", "kind": kind, "matched_correlation": round(math["matched_correlation"][(first, second)], 4),
              "matched_overlap": round(math["matched_overlap"][(first, second)], 4)}
             for first, second, kind, _ in math["seed_study_correlation"]["pairs"]]
    write_rows("attention_seed_pairs.csv", PAIR_COLUMNS, pairs)
    for name in ("correlation", "overlap"):
        study = math[f"seed_study_{name}"]
        print(f"matched {name}: " + ", ".join(f"{kind} {values['mean']:.3f}" for kind, values in study["summary"].items())
              + f"; same-loss minus other-loss {study['observed']:.4f}, permutation p = {study['p_value']:.3f}")
    print(f"Page: {write_page(results, probes)}")
