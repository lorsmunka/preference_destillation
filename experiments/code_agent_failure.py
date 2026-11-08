"""Coding student, where it fails (experiment plan 2026-11-06, #1).

The coding-agent student code-agent-kl-m-s1 (size m, pure KL to Gemma 4 12B, 10 passes) has its lowest test
loss at pass 2 and almost no exact scopes, and its attention never looks at the target lines. Does it locate
code at all, or does it only replay request -> scope patterns from the 16 training projects? Three parts, each
at pass 2 (lowest test loss) and pass 10 (the final pass), all on the CPU:

  a  teacher-forced on the run's own test records (the trainer's test set, so the "all records" row is the
     logged test CE): CE, KL and token accuracy, split into scope records and answer records (explain /
     found), and the scope records' tokens into file names (the name before .py / .html), line numbers
     and the rest (tags, extension, spaces, dashes, newlines)
  b  the student in its harness (both steps; its own scope narrows its own context; one retry each, as in
     Gemma's pass) on the 800 test requests: its scope against Gemma's and against the true places, which
     files it names (in the project shown, only in the training projects, made up), and what the second
     step did. Gemma's own pass (conversations/v1.jsonl) scored the same way is the reference row
  c  project swap: the student's first (greedy) scope for the same request text on 3 other versions of
     the same project (renamed and shuffled, so the true places move; only versions where every name the
     request mentions is called the same) and on a version of another test project. Does the scope change
     with the context, and does it follow the true places of the version shown or stay on the lines that
     were right in the request's own version?

    python experiments/code_agent_failure.py     # everything, ~2-2.5 h of CPU; rows -> results/code_agent_failure.csv
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only: the local GPU is not used

import json
import re
import sys
import time
from collections import Counter, defaultdict

from _common import write_rows

import torch
import torch.nn.functional as F

from library import RunStore
from library.agent.runner import StudentAgentModel, run_sessions
from library.domain import get_domain
from library.domain.code_agent.checks import (SCOPE_LINE, canonical_scope, check_scope, lines_of, parse_scope,
                                              score_scope, usable_ranges)
from library.domain.code_agent.harness import SCOPE_PREFIX, CodeAgentSession, version_cache
from library.domain.code_agent.project import load_base, project_list, symbol_lines
from library.domain.code_agent.requests import TEST_PROJECTS, VERSIONS, read_items
from library.model.student import StudentModel
from library.shared.batch_files import read_batch
from library.training.batch_handler import TEST, select_records
from library.training.example_encoder import ExampleEncoder
from library.training.losses import DistillationLoss

RUN_NAME = "code-agent-kl-m-s1"
CHECKPOINTS = {"pass 2": 1, "pass 10": 9}   # label -> n of checkpoint_epoch_<n>.pt
OTHER_VERSIONS = 3          # swapped versions per request in part c
TORCH_THREADS = 8
CHUNK = 100                 # requests per progress line
RESULTS_FILE = "code_agent_failure.csv"
RECORD_SPLITS = {"scope": "scope records", "explain": "explain records", "find": "found records"}
TOKEN_SPLITS = ["scope: file-name tokens", "scope: line-number tokens", "scope: other tokens"]
REQUEST_SPLITS = {"all requests": lambda item: True, "find requests": lambda item: item["kind"] == "find",
                  "explain requests": lambda item: item["kind"] == "explain",
                  "by meaning": lambda item: item["source"] == "meaning", "by name": lambda item: item["source"] == "name"}
COLUMNS = [
    "row", "run", "checkpoint", "part", "split",
    # a: teacher-forced
    "records", "tokens", "test_ce", "test_kl", "token_accuracy", "teacher_entropy", "teacher_agreement",
    # b and c: scopes
    "requests", "scope_first_valid", "scope_usable", "distinct_scopes", "top_scope_share", "scope_in_training",
    "files_in_project", "files_in_original_only", "files_in_training_only", "files_made_up",
    "right_file", "right_file_original", "chance_right_file", "recall", "precision", "places_hit", "places",
    "exact_truth", "recall_original_truth", "precision_original_truth", "truth_overlap_with_original",
    "recall_vs_gemma", "precision_vs_gemma", "exact_gemma", "same_files_gemma",
    # b: the second step
    "ended_at_scope", "route_explain", "route_find", "route_unclear", "route_correct", "route_same_gemma",
    "explanation_pass", "explanations",
    # c: does the scope move with the context
    "versions_per_request", "identical_to_own", "same_files_as_own", "same_numbers_as_own", "identical_all_versions",
    "identical_other_request_same_version",
]


def share(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


# --- a: teacher-forced test records ---------------------------------------------------------------

def token_categories(record):
    """One split per target token: in a scope record "file-name", "line-number" or "other" tokens; in an
    answer record none (the record's own split covers it)."""
    tokens = [step["token"].replace("▁", " ") for step in record["steps"]]
    if record["turn_type"] != "scope":
        return [None] * len(tokens)
    text = "".join(tokens)
    character_kinds = [None] * len(text)
    for match in SCOPE_LINE.finditer(text):
        stem_end = match.start(1) + match.group(1).rfind(".")
        for position in range(match.start(1), stem_end):
            character_kinds[position] = "file"
        for group in (2, 3):
            if match.group(group):
                for position in range(match.start(group), match.end(group)):
                    character_kinds[position] = "number"
    categories, position = [], 0
    for token in tokens:
        kinds = character_kinds[position:position + len(token)]
        position += len(token)
        categories.append(TOKEN_SPLITS[1] if "number" in kinds else TOKEN_SPLITS[0] if "file" in kinds else TOKEN_SPLITS[2])
    return categories


def part_a(student, run, label):
    """CE, KL and token accuracy on the run's own test records, by record kind and token kind."""
    encoder = ExampleEncoder(student.model, "cpu")
    divergence = DistillationLoss("forward_kl")
    totals = defaultdict(Counter)
    _, test_files = run.split_files()   # the trainer's rule, from the run's info.json
    for path in test_files:
        for record in select_records(read_batch(path), TEST):
            if not record["steps"]:
                continue
            encoded = encoder.encode(record)
            start = len(encoded.sentence_tokens) - 1
            prediction = student.model(encoded.input_tensor)[0, start:start + encoded.num_steps]
            targets = encoded.target_indices_tensor
            teacher_log_probabilities = F.log_softmax(encoded.target_logits_tensor, dim=-1)
            per_token = {
                "ce": F.cross_entropy(prediction, targets, reduction="none"),
                "kl": divergence.divergence_per_token(F.log_softmax(prediction, dim=-1), teacher_log_probabilities.exp(),
                                                      teacher_log_probabilities),
                "correct": (prediction.argmax(dim=-1) == targets).float(),
                "teacher_entropy": -(teacher_log_probabilities.exp() * teacher_log_probabilities).sum(dim=-1),
                "teacher_agreement": (encoded.target_logits_tensor.argmax(dim=-1) == targets).float(),
            }
            per_token = {name: values.tolist() for name, values in per_token.items()}
            record_splits = ["all records", RECORD_SPLITS[record["turn_type"]]]
            for split in record_splits:
                totals[split]["records"] += 1
            for index, category in enumerate(token_categories(record)):
                for split in record_splits + ([category] if category else []):
                    totals[split]["tokens"] += 1
                    for name, values in per_token.items():
                        totals[split][name] += values[index]
    rows = []
    for split in ["all records", *RECORD_SPLITS.values(), *TOKEN_SPLITS]:
        total = totals[split]
        tokens = total["tokens"]
        rows.append({"row": f"{label}|a|{split}", "run": RUN_NAME, "checkpoint": label, "part": "a teacher-forced",
                     "split": split, "records": total["records"] or None, "tokens": tokens,
                     "test_ce": round(total["ce"] / tokens, 4), "test_kl": round(total["kl"] / tokens, 4),
                     "token_accuracy": round(total["correct"] / tokens, 4),
                     "teacher_entropy": round(total["teacher_entropy"] / tokens, 4),
                     "teacher_agreement": round(total["teacher_agreement"] / tokens, 4)})
    return rows


# --- shared: what a scope does ----------------------------------------------------------------------

PLACES = {}


def places_of(version):
    """symbol_lines of a version, remembered per version."""
    if version.id not in PLACES:
        PLACES[version.id] = symbol_lines(version.files)
    return PLACES[version.id]


def true_lines(truth, version):
    places = places_of(version)
    return set().union(*[lines_of({file_name: [places[(file_name, name)]]}) for file_name, name in truth])


def overlap(chosen, target):
    """(recall, precision) of the chosen lines against target lines."""
    common = len(chosen & target)
    return common / len(target) if target else None, common / len(chosen) if chosen else 0.0


def file_kind(file_name, shown, original, training_files):
    if file_name in shown.files:
        return "files_in_project"
    if original is not None and file_name in original.files:
        return "files_in_original_only"
    if file_name in training_files:
        return "files_in_training_only"
    return "files_made_up"


def scope_facts(block, shown, truth, training, original=None, original_truth=None, reference=None):
    """One scope block on the version `shown`: the files it names, its usable lines against the true places
    (`truth`, None when the version has none, e.g. another project), against the true places of the
    request's own version (`original`, `original_truth`, part c), and against a reference scope's lines
    (Gemma's, part b)."""
    ranges = usable_ranges(block, shown.files)
    chosen = lines_of(ranges)
    named = list(parse_scope(block))
    facts = {"block": block, "canonical": canonical_scope(block), "usable": bool(ranges), "named": named,
             "file_kinds": Counter(file_kind(name, shown, original, training["files"]) for name in named),
             "scope_in_training": canonical_scope(block) in training["scopes"]}
    if truth is not None:
        scored = score_scope(ranges, [tuple(place) for place in truth], places_of(shown))
        true_files = {file_name for file_name, _ in truth}
        facts.update(recall=scored["recall"], precision=scored["precision"], places_hit=scored["places_hit"],
                     places=scored["places"], exact_truth=chosen == true_lines(truth, shown),
                     right_file=bool(set(ranges) & true_files), chance_right_file=len(true_files) / len(shown.files))
    if original is not None:
        original_target = true_lines(original_truth, original)
        recall, precision = overlap(lines_of(usable_ranges(block, original.files)), original_target)
        facts.update(recall_original_truth=recall, precision_original_truth=precision,
                     right_file_original=bool(set(usable_ranges(block, original.files)) & {name for name, _ in original_truth}))
        if truth is not None:
            target = true_lines(truth, shown)
            facts["truth_overlap_with_original"] = len(target & original_target) / len(target | original_target)
    if reference is not None:
        recall, precision = overlap(chosen, reference)
        facts.update(recall_vs_gemma=recall, precision_vs_gemma=precision, exact_gemma=chosen == reference,
                     same_files_gemma=set(ranges) == {name for name, _ in reference})
    return facts


def scope_columns(facts_list):
    """The scope columns of a row, from the facts of its scopes."""
    canonical = Counter(facts["canonical"] for facts in facts_list)
    kinds = sum((facts["file_kinds"] for facts in facts_list), Counter())
    mentions = sum(kinds.values())
    columns = {"requests": len(facts_list), "scope_usable": share(facts["usable"] for facts in facts_list),
               "distinct_scopes": len(canonical), "top_scope_share": round(canonical.most_common(1)[0][1] / len(facts_list), 4),
               "scope_in_training": share(facts["scope_in_training"] for facts in facts_list)}
    places = sum(facts.get("places", 0) for facts in facts_list)
    if places:
        columns.update(places_hit=sum(facts["places_hit"] for facts in facts_list), places=places)
    for kind in ("files_in_project", "files_in_original_only", "files_in_training_only", "files_made_up"):
        columns[kind] = round(kinds[kind] / mentions, 4) if mentions else None
    for key in ("right_file", "right_file_original", "chance_right_file", "recall", "precision", "exact_truth",
                "recall_original_truth", "precision_original_truth", "truth_overlap_with_original",
                "recall_vs_gemma", "precision_vs_gemma", "exact_gemma", "same_files_gemma"):
        columns[key] = share(facts.get(key) for facts in facts_list)
    return columns


def training_reference(domain):
    """What the student saw in training: Gemma's scopes on the training projects (canonical strings) and every
    file name a training project's version can have (its base names and their renames)."""
    scopes = set()
    for line in open(domain.conversations_path, encoding="utf-8"):
        played = json.loads(line)
        if played["split"] == "train" and played["turn"].get("scope"):
            scopes.add(canonical_scope(played["turn"]["scope"]))
    files = set()
    for project in project_list():
        if project in TEST_PROJECTS:
            continue
        base = load_base(project)
        for file_name in base["files"]:
            stem, extension = file_name.rsplit(".", 1)
            files |= {f"{name}.{extension}" for name in [stem] + base["renames"].get(stem, [])}
    return {"scopes": scopes, "files": files}


# --- b: the student in its harness ------------------------------------------------------------------

def first_scope(turn):
    """The first scope the model wrote (the greedy one): the rejected one when it was asked again."""
    rejected = [entry["output"] for entry in turn.get("rejected", []) if entry["step"] == "scope"]
    return rejected[0] if rejected else turn.get("scope", "")


def gemma_turns(domain):
    turns = {}
    for line in open(domain.conversations_path, encoding="utf-8"):
        played = json.loads(line)
        if played["split"] == "test":
            turns[played["item_id"]] = played["turn"]
    return turns


def gemma_lines(turn):
    return lines_of({name: [tuple(span) for span in spans] for name, spans in (turn.get("ranges") or {}).items()})


def play(student, domain, items, versions):
    """Every item played by the student in the harness, both steps; the retries sample (seeded)."""
    torch.manual_seed(0)
    model = StudentAgentModel(student, domain.max_steps)
    turns = []
    for start in range(0, len(items), CHUNK):
        sessions = [CodeAgentSession(item, versions(item["project"], item["version"])) for item in items[start:start + CHUNK]]
        run_sessions(model, sessions)
        turns += [session.turn for session in sessions]
        print(f"  harness: {start + len(sessions)}/{len(items)} requests", flush=True)
    return turns


def harness_rows(label, items, turns, versions, training, gemma=None):
    """Part b rows: the scope the harness went on with, and the second step. `gemma` (Gemma's turns by item)
    adds the comparison with Gemma's scope and decision."""
    rows = []
    for split, belongs in REQUEST_SPLITS.items():
        chosen = [(item, turn) for item, turn in zip(items, turns) if belongs(item)]
        facts_list, routes = [], []
        for item, turn in chosen:
            version = versions(item["project"], item["version"])
            reference = gemma_lines(gemma[item["id"]]) if gemma else None
            facts_list.append(scope_facts(turn.get("scope", ""), version, item["truth"], training, reference=reference))
            routes.append(turn.get("route") or "ended at scope")
        explained = [turn for (_, turn), route in zip(chosen, routes) if route == "explain"]
        row = {"row": f"{label}|b|{split}", "run": "gemma" if label == "gemma" else RUN_NAME, "checkpoint": label,
               "part": "b harness", "split": split, **scope_columns(facts_list),
               "scope_first_valid": share(check_scope(first_scope(turn), versions(item["project"], item["version"]).files) is None
                                          for item, turn in chosen),
               "ended_at_scope": share(route == "ended at scope" for route in routes),
               "route_explain": share(route == "explain" for route in routes),
               "route_find": share(route == "find" for route in routes),
               "route_unclear": share(route == "unclear" for route in routes),
               "route_correct": share(bool(turn.get("route_correct")) for _, turn in chosen),
               "explanation_pass": share(turn.get("check") is None for turn in explained), "explanations": len(explained)}
        if gemma:
            row["route_same_gemma"] = share(route == (gemma[item["id"]].get("route") or "ended at scope")
                                            for (item, _), route in zip(chosen, routes))
        rows.append(row)
    return rows


# --- c: project swap ----------------------------------------------------------------------------------

def message_fits(message, renameable, original, other):
    """Whether a request written for `original` means the same on `other`: no name it mentions is called
    otherwise there."""
    for name in renameable:
        here, there = original.renames.get(name, name), other.renames.get(name, name)
        if here != there and re.search(rf"(?<![\w-]){re.escape(here)}(?![\w-])", message):
            return False
    return True


def moved_truth(truth, original, other):
    """The true places of a request on `original`, as `other` calls them (back to the base names, then on)."""
    base_names = {renamed: name for name, renamed in original.renames.items()}
    moved = []
    for file_name, name in truth:
        stem, extension = file_name.rsplit(".", 1)
        base_name = ".".join(base_names.get(part, part) for part in name.split("."))
        moved.append(list(other.place(f"{base_names.get(stem, stem)}.{extension}:{base_name}")))
    return moved


def swap_versions(item, versions, renameable):
    """Up to OTHER_VERSIONS other versions of the item's project on which its request means the same."""
    original = versions(item["project"], item["version"])
    chosen = []
    for offset in range(1, VERSIONS["test"]):
        other = versions(item["project"], (item["version"] + offset) % VERSIONS["test"])
        if message_fits(item["message"], renameable, original, other):
            chosen.append(other)
        if len(chosen) == OTHER_VERSIONS:
            break
    return chosen


def other_project(item, versions):
    """The same version number of the next test project."""
    index = TEST_PROJECTS.index(item["project"])
    return versions(TEST_PROJECTS[(index + 1) % len(TEST_PROJECTS)], item["version"])


def first_scopes(student, domain, items_and_versions):
    """The student's first (greedy) scope for each (item, version): the harness's first request, nothing more."""
    model = StudentAgentModel(student, domain.max_steps)
    blocks = []
    for start in range(0, len(items_and_versions), CHUNK):
        requests = [CodeAgentSession(item, version).pending for item, version in items_and_versions[start:start + CHUNK]]
        blocks += [SCOPE_PREFIX + text for text in model.generate(requests)]
        print(f"  swap: {start + len(requests)}/{len(items_and_versions)} scopes", flush=True)
    return blocks


def swap_rows(label, student, domain, items, turns, versions, training):
    renameable = {project: list(load_base(project)["renames"]) for project in TEST_PROJECTS}
    plays, skipped = [], 0   # (item index, shown version, truth there or None, kind)
    for index, item in enumerate(items):
        original = versions(item["project"], item["version"])
        for other in swap_versions(item, versions, renameable[item["project"]]):
            truth = moved_truth(item["truth"], original, other)
            if all(tuple(place) in places_of(other) for place in truth):
                plays.append((index, other, truth, "other versions"))
            else:
                skipped += 1
        plays.append((index, other_project(item, versions), None, "other project"))
    if skipped:
        print(f"  {skipped} swapped versions skipped: a true place was not found again")
    blocks = first_scopes(student, domain, [(items[index], version) for index, version, _, _ in plays])

    own_blocks = [first_scope(turn) for turn in turns]
    facts_by_kind = defaultdict(list)
    canonical_by_item = defaultdict(set)
    for index, item in enumerate(items):
        version = versions(item["project"], item["version"])
        facts = scope_facts(own_blocks[index], version, item["truth"], training, original=version, original_truth=item["truth"])
        facts_by_kind["own version"].append({**facts, "index": index})
        canonical_by_item[index].add(facts["canonical"])
    for (index, version, truth, kind), block in zip(plays, blocks):
        item = items[index]
        original = versions(item["project"], item["version"])
        facts = scope_facts(block, version, truth, training, original=original, original_truth=item["truth"])
        facts_by_kind[kind].append({**facts, "index": index})
        if kind == "other versions":
            canonical_by_item[index].add(facts["canonical"])

    own = facts_by_kind["own version"]
    rows = []
    for kind in ("own version", "other versions", "other project"):
        facts_list = facts_by_kind[kind]
        row = {"row": f"{label}|c|{kind}", "run": RUN_NAME, "checkpoint": label, "part": "c project swap", "split": kind,
               **scope_columns(facts_list),
               "identical_to_own": share(facts["canonical"] == own[facts["index"]]["canonical"] for facts in facts_list),
               "same_files_as_own": share(set(facts["named"]) == set(own[facts["index"]]["named"]) for facts in facts_list),
               "same_numbers_as_own": share(scope_numbers(facts["block"]) == scope_numbers(own[facts["index"]]["block"])
                                            for facts in facts_list)}
        if kind == "own version":
            row["identical_other_request_same_version"] = identical_within_versions(items, facts_list)
        if kind == "other versions":
            per_item = Counter(facts["index"] for facts in facts_list)
            row["versions_per_request"] = round(sum(per_item.values()) / len(items), 3)
            row["identical_all_versions"] = share(len(canonical_by_item[index]) == 1 for index in per_item)
        rows.append(row)
    show_swap_examples(items, own_blocks, plays, blocks)
    return rows


def identical_within_versions(items, facts_list):
    """The share of pairs of different requests on the same version whose scopes are identical (the control
    for identical_to_own: same request, other version)."""
    by_version = defaultdict(list)
    for facts in facts_list:
        item = items[facts["index"]]
        by_version[(item["project"], item["version"])].append(facts["canonical"])
    pairs = [first == second for scopes in by_version.values()
             for position, first in enumerate(scopes) for second in scopes[position + 1:]]
    return share(pairs)


def scope_numbers(block):
    """A scope's line ranges without their files, sorted."""
    return sorted(span for spans in parse_scope(block).values() for span in spans)


def show_swap_examples(items, own_blocks, plays, blocks, count=4):
    for index in range(0, len(items), max(1, len(items) // count)):
        item = items[index]
        print(f"  [{item['id']}] {item['message']}  truth {item['truth']}")
        print(f"     own version: {canonical_scope(own_blocks[index])}")
        for (play_index, version, truth, kind), block in zip(plays, blocks):
            if play_index == index:
                print(f"     {kind} {version.id}: {canonical_scope(block)}   (truth there: {truth})")


# --- the whole thing -------------------------------------------------------------------------------------

def print_rows(rows):
    for row in rows:
        values = {key: value for key, value in row.items()
                  if key not in ("row", "run", "checkpoint", "part", "split") and value is not None}
        print(f"  {row['checkpoint']:<8} {row['part']:<17} {row['split']:<26} " + ", ".join(f"{key} {value}" for key, value in values.items()))


def main():
    torch.set_num_threads(TORCH_THREADS)
    torch.set_grad_enabled(False)
    store = RunStore()
    run = store.get(RUN_NAME)
    domain = get_domain(run.domain)
    items = [item for item in read_items() if item["split"] == "test"]
    versions = version_cache()
    training = training_reference(domain)
    gemma = gemma_turns(domain)
    print(f"{len(items)} test requests, {len(gemma)} played by Gemma; {len(training['scopes'])} distinct training scopes, "
          f"{len(training['files'])} training file names")

    gemma_rows = harness_rows("gemma", items, [gemma[item["id"]] for item in items], versions, training)
    write_rows(RESULTS_FILE, COLUMNS, gemma_rows)
    all_rows = list(gemma_rows)
    for label, epoch in CHECKPOINTS.items():
        _, checkpoint_path = store.resolve_checkpoint(run.name, str(epoch))
        student = StudentModel.from_run(run, checkpoint_path, device="cpu", domain=domain)
        print(f"\n=== {label}: {checkpoint_path.name} ===", flush=True)

        started = time.time()
        rows = part_a(student, run, label)
        write_rows(RESULTS_FILE, COLUMNS, rows)
        print(f"  a: {time.time() - started:.0f} s", flush=True)
        all_rows += rows

        started = time.time()
        turns = play(student, domain, items, versions)
        rows = harness_rows(label, items, turns, versions, training, gemma=gemma)
        write_rows(RESULTS_FILE, COLUMNS, rows)
        print(f"  b: {time.time() - started:.0f} s; most common scopes:", flush=True)
        for scope, count in Counter(canonical_scope(turn.get("scope", "")) for turn in turns).most_common(8):
            print(f"     {count:>4}  {scope}")
        all_rows += rows

        started = time.time()
        rows = swap_rows(label, student, domain, items, turns, versions, training)
        write_rows(RESULTS_FILE, COLUMNS, rows)
        print(f"  c: {time.time() - started:.0f} s", flush=True)
        all_rows += rows
        del student
    print()
    print_rows(all_rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)   # every line reaches the log, even if the run is stopped
    main()
