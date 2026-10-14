"""Does a trained slice student use its input? (CPU, 2026-10-14) Teacher-forced accuracy on Gemma's replies
with the true conversation vs another record's conversation (same turn type), on train and test
records; separately at digit positions (numbers the student must take from the problem).
Result on the trial students (slice-kd-s1 / slice-ce-s1): test digits 31% with the right
conversation vs 24% with another one (train 38% vs 25%): the input is barely used.
With one update per batch (slice-*-batched-s1): test 27% vs 22%, train 35% vs 22%: no better.

    python experiments/input_use_check.py   # after experiments/slice_trial.py"""
import random, sys
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from library import RunStore
from library.model.student import StudentModel
from library.domain.math_tutor.slice import SliceDomain
from library.shared.batch_files import batch_paths, read_batch
from library.shared import get_output_dir
from library.training.example_encoder import ExampleEncoder

domain = SliceDomain()
store = RunStore()
examples = [example for p in batch_paths(get_output_dir(domain.name, "google/gemma-4-12B-it")) for example in read_batch(p)]
random_generator = random.Random(0)

def scores(model, encoder, example):
    encoded = encoder.encode(example)
    with torch.no_grad():
        logits = model(encoded.input_tensor)[0]
    start = len(encoded.sentence_tokens) - 1
    predicted = logits[start:start + encoded.num_steps].argmax(-1)
    correct = (predicted == encoded.target_indices_tensor).tolist()
    digit = [step["token"].strip("▁").isdigit() for step in example["steps"]]
    return correct, digit

RUN_NAMES = ["slice-kd-s1", "slice-ce-s1",                   # one update per example (before 2026-10-14)
             "slice-kd-batched-s1", "slice-ce-batched-s1"]   # one update per batch
for arm in [name for name in RUN_NAMES if (store.runs_dir / name / "info.json").exists()]:
    random_generator = random.Random(0)   # the same records and swaps for every run
    run = store.get(arm)
    _, path = store.resolve_checkpoint(run.name, "latest")
    student = StudentModel.from_run(run, path, device="cpu", domain=domain)
    model = student.model.eval()
    encoder = ExampleEncoder(model, "cpu")
    for split in ("train", "test"):
        for turn_type in ("solve", "explain"):
            pool = [e for e in examples if e["split"] == split and e["turn_type"] == turn_type]
            sample = random_generator.sample(pool, min(60, len(pool)))
            totals = {"true": [0, 0, 0, 0], "swapped": [0, 0, 0, 0]}   # correct, steps, digit correct, digits
            for example in sample:
                other = random_generator.choice([e for e in pool if e["item_id"] != example["item_id"]])
                for name, sentence in (("true", example["sentence"]), ("swapped", other["sentence"])):
                    correct, digit = scores(model, encoder, {**example, "sentence": sentence})
                    totals[name][0] += sum(correct); totals[name][1] += len(correct)
                    totals[name][2] += sum(c for c, d in zip(correct, digit) if d); totals[name][3] += sum(digit)
            line = ", ".join(f"{name}: all {t[0]/t[1]:.3f}, digits {t[2]/max(1,t[3]):.3f} ({t[3]})" for name, t in totals.items())
            print(f"{arm} {split:5} {turn_type:7} n={len(sample)}  {line}", flush=True)
