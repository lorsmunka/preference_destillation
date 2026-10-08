"""E1 mechanism + E4 calibration — post-hoc probes over the vocab-sweep checkpoints.

Three measurements the training logs cannot give, taken from each trained checkpoint
without changing the framework:

1. HEAD-VS-BODY GRADIENT FRACTION (board A2, metric from 2603.10145). Runs the real
   distillation loss backward and reports how much of the total gradient norm lands on
   the output projection versus the transformer body. The bottleneck claim predicts a
   smaller output vocabulary lets MORE gradient reach the body.
2. HIDDEN-STATE EFFECTIVE RANK (board A1 degeneracy). Singular-value spectrum of the
   final hidden states, summarised as participation ratio. Degenerate/anisotropic
   representations collapse the spectrum onto few directions.
3. EXPECTED CALIBRATION ERROR (board C1/C2). Ten-bin ECE over teacher-forced next-token
   predictions — the cheap novel eval axis, measured against the vocab axis.

    python experiments/probe_bottleneck.py

Reads whatever `ideaA-*` runs have completed; safe to re-run as more finish. The probe
examples are the first PROBE_EXAMPLES of the fixed test tail (the trainer's split rule).
"""

import math

import torch
import torch.nn.functional as F

from _common import write_rows

from library import RunStore
from library.model.student import StudentModel
from library.shared.batch_files import read_batch
from library.training.batch_handler import TEST, select_records

PROBE_EXAMPLES = 20        # examples for ECE + isotropy
GRADIENT_EXAMPLES = 8      # backward passes (one per example, gradients accumulated)
GRADIENT_SEED = 0          # dropout masks of the gradient probe
CALIBRATION_BINS = 10


def test_examples(run, limit):
    """The first `limit` examples of the run's test set, by the trainer's own split rule
    (Run.split_files): the fixed test tail, which no run trained on."""
    examples = []
    for path in run.split_files()[1]:
        for example in select_records(read_batch(path), TEST):
            examples.append(example)
            if len(examples) >= limit:
                return examples
    return examples


def parameter_groups(model):
    """Split parameters into the head (output projection) and the body that feeds it.
    `input_embedding` is reported separately: it dominates the parameter count here and
    would otherwise swamp the body's gradient norm."""
    return {
        "head": [p for name, p in model.named_parameters() if name.startswith("output_projection")],
        "body": [p for name, p in model.named_parameters()
                 if name.startswith("transformer_layers") or name.startswith("layer_norm")],
        "embedding": [p for name, p in model.named_parameters() if name.startswith("input_embedding")],
    }


def group_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().float().norm().item() ** 2
    return math.sqrt(total)


def gradient_fraction(student, examples):
    """Accumulate the distillation gradient over a few examples, then report the share of
    total gradient norm sitting on the head. Dropout is on (train mode, as in training), so
    the dropout masks are seeded: the same checkpoint always gives the same fraction."""
    model = student.model
    model.zero_grad(set_to_none=True)
    model.train()
    torch.manual_seed(GRADIENT_SEED)

    used = 0
    for example in examples[:GRADIENT_EXAMPLES]:
        steps = example.get("steps", [])
        if not steps:
            continue
        sentence_ids = student.tokenizer.encode(
            student.domain.student_prompt(example["sentence"]), add_special_tokens=False)
        token_ids = [student._token_id(step["token"]) for step in steps]
        remapped = model.remap_input_tokens(sentence_ids + token_ids[:-1])

        tensor = torch.tensor([remapped], dtype=torch.long, device=student.device)
        logits = model(tensor)[0]
        start = len(sentence_ids) - 1
        prediction_logits = logits[start: start + len(steps), :]

        teacher_logits = torch.tensor([s["logits"][: student.vocab_size] for s in steps],
                                      dtype=torch.float32, device=student.device)
        loss = F.kl_div(F.log_softmax(prediction_logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1), reduction="batchmean")
        loss.backward()
        used += 1

    if used == 0:
        return None
    groups = parameter_groups(model)
    norms = {name: group_norm(params) for name, params in groups.items()}
    total = norms["head"] + norms["body"] + norms["embedding"]
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "grad_norm_head": norms["head"],
        "grad_norm_body": norms["body"],
        "grad_norm_embedding": norms["embedding"],
        "head_gradient_fraction": norms["head"] / total if total else None,
        "body_gradient_fraction": norms["body"] / total if total else None,
        "gradient_examples": used,
    }


def effective_rank(hidden_states):
    """Participation ratio of the singular-value spectrum: (sum s)^2 / sum s^2.
    Equals the dimension count for a perfectly isotropic cloud, 1 for a rank-1 one."""
    if hidden_states.shape[0] < 2:
        return None
    centred = hidden_states - hidden_states.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centred.float())
    total = singular_values.sum().item()
    squared = (singular_values ** 2).sum().item()
    return (total ** 2) / squared if squared else None


def calibration_and_rank(student, examples):
    """ECE over teacher-forced predictions + effective rank of the final hidden states."""
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach()[0])  # [seq, hidden]

    handle = student.model.layer_norm.register_forward_hook(hook)

    confidences, correct = [], []
    with torch.no_grad():
        for example in examples:
            forced = student.teacher_forced(example)
            if forced is None:
                continue
            probabilities = F.softmax(forced["prediction_logits"], dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            confidences.append(confidence)
            correct.append((predicted == forced["target_indices"]).float())

    handle.remove()
    if not confidences:
        return None

    confidence_tensor = torch.cat(confidences)
    correct_tensor = torch.cat(correct)

    ece = 0.0
    total = confidence_tensor.numel()
    for index in range(CALIBRATION_BINS):
        low = index / CALIBRATION_BINS
        high = (index + 1) / CALIBRATION_BINS
        in_bin = (confidence_tensor > low) & (confidence_tensor <= high)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        bin_confidence = confidence_tensor[in_bin].mean().item()
        bin_accuracy = correct_tensor[in_bin].mean().item()
        ece += (count / total) * abs(bin_confidence - bin_accuracy)

    hidden = torch.cat(captured, dim=0) if captured else None
    return {
        "ece": ece,
        "mean_confidence": confidence_tensor.mean().item(),
        "token_accuracy": correct_tensor.mean().item(),
        "overconfidence": confidence_tensor.mean().item() - correct_tensor.mean().item(),
        "effective_rank": effective_rank(hidden) if hidden is not None else None,
        "hidden_dim": hidden.shape[1] if hidden is not None else None,
        "probe_steps": total,
    }


COLUMNS = ["run", "output_vocab_size", "seed", "ece", "mean_confidence", "token_accuracy",
           "overconfidence", "effective_rank", "hidden_dim", "probe_steps",
           "head_gradient_fraction", "body_gradient_fraction", "grad_norm_head",
           "grad_norm_body", "grad_norm_embedding", "gradient_examples"]


def main():
    store = RunStore()
    runs = [r for r in store.all()
            if r.name.startswith("ideaA-") and r.status == "completed"]
    if not runs:
        print("No completed ideaA-* runs yet.")
        return
    print(f"Probing {len(runs)} runs\n")

    rows = []
    for run in runs:
        try:
            _label, checkpoint_path = store.resolve_checkpoint(run.name, "latest")
        except FileNotFoundError:
            print(f"  no checkpoint: {run.name}")
            continue
        student = StudentModel.from_run(run, checkpoint_path)
        examples = test_examples(run, PROBE_EXAMPLES)

        row = {"run": run.name,
               "output_vocab_size": run.output_vocab_size,
               "seed": run.name.rsplit("-s", 1)[-1]}
        measured = calibration_and_rank(student, examples)
        if measured:
            row.update(measured)
        gradients = gradient_fraction(student, examples)
        if gradients:
            row.update(gradients)
        rows.append({key: row.get(key) for key in COLUMNS})
        print(f"  {run.name:<28} V={run.output_vocab_size:<5} "
              f"ECE={row.get('ece', 0):.4f} rank={row.get('effective_rank') or 0:.2f} "
              f"head_grad={row.get('head_gradient_fraction') or 0:.4f}")

    write_rows("ideaA_bottleneck_probe.csv", COLUMNS, rows)


if __name__ == "__main__":
    main()
