"""Resume correctness of the real TrainingRunner/Trainer, CPU, tiny model, real reddit
batches read-only. Runs land in this scratch folder (get_training_run_dir monkeypatched)."""
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, r"D:\preference_destillation")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(SCRATCH, "fake_runs")
os.chdir(SCRATCH)

import torch
import library.training.training_runner as training_runner_module
import library.training.trainer as trainer_module
from library import TrainingRun
from library.domain import REDDIT_SENTIMENT

training_runner_module.get_training_run_dir = lambda name: os.path.join(RUNS_ROOT, name)
trainer_module.plot_training_logs = lambda *a, **k: None
training_runner_module.plot_training_logs = lambda *a, **k: None


class FakeExit:
    """Requests an exit after the Nth call of check_exit (one call per training batch)."""
    def __init__(self, exit_after_calls=None):
        self.calls = 0
        self.exit_after_calls = exit_after_calls

    def check_exit(self):
        self.calls += 1
        return self.exit_after_calls is not None and self.calls >= self.exit_after_calls

    def stop(self):
        pass


def spec(name):
    return TrainingRun(domain=REDDIT_SENTIMENT, run_name=name, hidden_dim=8, num_layers=1, num_heads=1,
                       dropout=0.0, epoch_count=2, batch_size=32, learning_rate=0.0015,
                       lr_warmup_ratio=0.1, max_training_examples=160, training_test_ratio=0.8,
                       kl_ratio_start=0.9, kl_ratio_end=0.1)


def weights(name, file="checkpoint_epoch_1.pt"):
    path = os.path.join(RUNS_ROOT, name, "checkpoints", file)
    return torch.load(path, weights_only=True)["model_state_dict"] if os.path.exists(path) else None


def run(name, exit_after=None):
    torch.manual_seed(1)
    return training_runner_module.TrainingRunner(spec(name), FakeExit(exit_after)).run()


def max_difference(a, b):
    return max((a[key].float() - b[key].float()).abs().max().item() for key in a if "rotary" not in key)


shutil.rmtree(RUNS_ROOT, ignore_errors=True)
# 160 examples -> 5 batches -> 4 train, 1 test; 2 epochs -> 8 training batches
run("straight")
reference = weights("straight")

# exit after batch 2 of epoch 0, then resume
print("stopped:", run("mid", exit_after=2))
print("resumed:", run("mid"))
print("mid-epoch resume max |dw| vs straight:", max_difference(reference, weights("mid")))

# exit after the LAST batch of epoch 0 (4th check)
print("stopped:", run("edge", exit_after=4))
print("resumed:", run("edge"))
print("edge resume max |dw| vs straight:", max_difference(reference, weights("edge")))
print("edge checkpoints:", sorted(os.listdir(os.path.join(RUNS_ROOT, "edge", "checkpoints"))))
with open(os.path.join(RUNS_ROOT, "edge", "logs", "training.jsonl")) as file:
    records = [json.loads(line) for line in file]
print("edge train_epoch records:", [r["epoch"] for r in records if r["type"] == "train_epoch"],
      " eval_epoch records:", [r["epoch"] for r in records if r["type"] == "eval_epoch"])

# stale temp checkpoint: graceful stop in epoch 0, resume to finish epoch 0, then a hard crash
# in epoch 1 (simulated by raising inside the batch loop) and a restart.
print("stopped:", run("stale", exit_after=1))
original_process_batch = trainer_module.Trainer._process_batch
def crashing_process_batch(self, batch_index, epoch):
    if epoch == 1 and batch_index == 2:
        raise RuntimeError("simulated hard crash (exception / power loss)")
    return original_process_batch(self, batch_index, epoch)
trainer_module.Trainer._process_batch = crashing_process_batch
try:
    run("stale")
except RuntimeError as error:
    print("crashed:", error)
trainer_module.Trainer._process_batch = original_process_batch
with open(os.path.join(RUNS_ROOT, "stale", "logs", "state.json")) as file:
    print("state after crash:", json.load(file))
print("restart:", run("stale"))
print("stale-restart max |dw| vs straight:", max_difference(reference, weights("stale")))
with open(os.path.join(RUNS_ROOT, "stale", "info.json")) as file:
    print("stale status:", json.load(file)["status"])
