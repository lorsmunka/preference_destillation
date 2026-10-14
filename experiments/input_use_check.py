"""Does a trained slice student use its input? (CPU, 2026-10-14) Teacher-forced accuracy on Gemma's replies
with the true conversation vs another record's conversation (same turn type), on train and test
records; separately at digit positions (numbers the student must take from the problem).
Result on the trial students (slice-kd-s1 / slice-ce-s1): test digits 31% with the right
conversation vs 24% with another one (train 38% vs 25%): the input is barely used.
With one update per batch (slice-*-batched-s1): test 27% vs 22%, train 35% vs 22%: no better.
(Both on the trial's teacher data of 224 conversations; experiments/slice_curve.py reuses input_use.)

    python experiments/input_use_check.py   # after experiments/slice_trial.py"""
import random
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from library import RunStore
from library.domain.math_tutor.slice import SliceDomain
from library.model.student import StudentModel
from library.shared import get_output_dir
from library.shared.batch_files import batch_paths, read_batch, read_batch_meta
from library.training.example_encoder import ExampleEncoder

TEACHER = "google/gemma-4-12B-it"
SAMPLE_SIZE = 60   # records per split and turn type
RUN_NAMES = ["slice-kd-s1", "slice-ce-s1",                   # one update per example (before 2026-10-14)
             "slice-kd-batched-s1", "slice-ce-batched-s1"]   # one update per batch

RecordEntry = namedtuple("RecordEntry", "path position split turn_type item_id sentence")


def record_index():
    """Where every record of the slice's teacher data is, read from the batch files' metadata only
    (the full teacher data does not fit in memory)."""
    entries = []
    for path in batch_paths(get_output_dir(SliceDomain().name, TEACHER)):
        meta = read_batch_meta(path)
        entries += [RecordEntry(path, position, *fields) for position, fields in
                    enumerate(zip(meta["split"], meta["turn_type"], meta["item_id"], meta["sentences"]))]
    return entries


def sample_pairs(entries, split, turn_type, seed=0, sample_size=SAMPLE_SIZE):
    """`sample_size` records of one split and turn type, each paired with the conversation of
    another problem's record of the same type. Only the batch files holding them are read."""
    random_generator = random.Random(seed)
    pool = [entry for entry in entries if entry.split == split and entry.turn_type == turn_type]
    chosen = random_generator.sample(pool, min(sample_size, len(pool)))
    others = [random_generator.choice([entry for entry in pool if entry.item_id != record.item_id]).sentence
              for record in chosen]
    positions_by_path = defaultdict(list)
    for number, record in enumerate(chosen):
        positions_by_path[record.path].append((number, record.position))
    examples = [None] * len(chosen)
    for path, positions in positions_by_path.items():
        records = read_batch(path)
        for number, position in positions:
            examples[number] = records[position]
    return list(zip(examples, others))


def scores(model, encoder, example):
    encoded = encoder.encode(example)
    with torch.no_grad():
        logits = model(encoded.input_tensor)[0]
    start = len(encoded.sentence_tokens) - 1
    predicted = logits[start:start + encoded.num_steps].argmax(-1)
    correct = (predicted == encoded.target_indices_tensor).tolist()
    digit = [step["token"].strip("▁").isdigit() for step in example["steps"]]
    return correct, digit


def input_use(run_name, pairs):
    """Teacher-forced accuracy of one run on the sampled replies, with their own conversation and
    with another one: {"own": {...}, "other": {...}}, each with all-token and digit accuracy."""
    store = RunStore()
    run = store.get(run_name)
    _, path = store.resolve_checkpoint(run.name, "latest")
    model = StudentModel.from_run(run, path, device="cpu", domain=SliceDomain()).model.eval()
    encoder = ExampleEncoder(model, "cpu")
    totals = {"own": [0, 0, 0, 0], "other": [0, 0, 0, 0]}   # correct, steps, digits correct, digits
    for example, other_sentence in pairs:
        for name, sentence in (("own", example["sentence"]), ("other", other_sentence)):
            correct, digit = scores(model, encoder, {**example, "sentence": sentence})
            totals[name][0] += sum(correct)
            totals[name][1] += len(correct)
            totals[name][2] += sum(c for c, d in zip(correct, digit) if d)
            totals[name][3] += sum(digit)
    return {name: {"all": round(t[0] / max(1, t[1]), 3), "digits": round(t[2] / max(1, t[3]), 3), "digit_count": t[3]}
            for name, t in totals.items()}


def main():
    store = RunStore()
    entries = record_index()
    groups = [(split, turn_type) for split in ("train", "test") for turn_type in ("solve", "explain")]
    pairs = {group: sample_pairs(entries, *group) for group in groups}   # the same records and swaps for every run
    for run_name in [name for name in RUN_NAMES if (store.runs_dir / name / "info.json").exists()]:
        for split, turn_type in groups:
            result = input_use(run_name, pairs[(split, turn_type)])
            line = ", ".join(f"{name}: all {value['all']:.3f}, digits {value['digits']:.3f} ({value['digit_count']})"
                             for name, value in result.items())
            print(f"{run_name} {split:5} {turn_type:7} n={len(pairs[(split, turn_type)])}  {line}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
