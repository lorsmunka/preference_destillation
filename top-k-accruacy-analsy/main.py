from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from shared import PROMPT_DELIMITER, get_batches_dir, get_device, load_input_vocabulary
from training.batch_handler import BatchHandler
from training.model import Transformer


DEFAULT_CONFIG = {
    "top_k": 20,
    "split": "test",
    "checkpoint": "latest",
    "max_batches": None,
    "max_examples": None,
    "models": [],
}

VALID_SPLITS = {"train", "test", "all"}


@dataclass
class MetricTotals:
    top_k_hits: int = 0
    top_k_overlap_sum: float = 0.0
    target_rank_sum: float = 0.0
    total_steps: int = 0
    total_examples: int = 0
    total_batches: int = 0

    def as_metrics(self) -> Dict[str, float]:
        if self.total_steps == 0:
            return {
                "top_k_accuracy": 0.0,
                "teacher_student_top_k_overlap": 0.0,
                "mean_target_rank": 0.0,
            }

        return {
            "top_k_accuracy": self.top_k_hits / self.total_steps,
            "teacher_student_top_k_overlap": self.top_k_overlap_sum / self.total_steps,
            "mean_target_rank": self.target_rank_sum / self.total_steps,
        }


def main() -> None:
    os.chdir(REPOSITORY_ROOT)

    parser = argparse.ArgumentParser(
        description="Measure top-k student/teacher metrics for training runs."
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.json"),
        help="Path to the analysis config JSON.",
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Run the config exactly as saved, without interactive prompts.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Update the config interactively, then exit without running analysis.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    if not args.from_config:
        config = configure_interactively(config)
        save_config(config_path, config)
        if args.setup_only:
            print(f"\nSaved config: {config_path}")
            return

        run_now = prompt_yes_no("\nRun analysis now?", default=True)
        if not run_now:
            print(f"Saved config: {config_path}")
            return

    validate_config(config)
    results = run_analysis(config)
    write_outputs(config, results)


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = json.load(file)

    config = dict(DEFAULT_CONFIG)
    config.update(loaded_config)
    config["models"] = normalize_model_entries(config.get("models", []))
    return config


def save_config(config_path: Path, config: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")


def normalize_model_entries(entries: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized = []
    seen_run_names = set()

    for entry in entries:
        if isinstance(entry, str):
            model_entry = {"run_name": entry}
        elif isinstance(entry, dict):
            model_entry = dict(entry)
        else:
            continue

        run_name = model_entry.get("run_name")
        if not run_name or run_name in seen_run_names:
            continue

        seen_run_names.add(run_name)
        normalized.append(model_entry)

    return normalized


def configure_interactively(config: Dict[str, Any]) -> Dict[str, Any]:
    print("=== Top-k Accuracy Analysis Setup ===\n")
    print("Press enter to keep the value shown in brackets.\n")

    config["top_k"] = prompt_integer("Top-k", config.get("top_k", 20), minimum=1)
    config["split"] = prompt_choice("Corpus split", config.get("split", "test"), VALID_SPLITS)
    config["checkpoint"] = prompt_text("Checkpoint", config.get("checkpoint", "latest"))
    config["max_batches"] = prompt_optional_integer(
        "Max batches (blank or none means all selected split batches)",
        config.get("max_batches"),
        minimum=1,
    )
    config["max_examples"] = prompt_optional_integer(
        "Max examples (blank or none means all selected examples)",
        config.get("max_examples"),
        minimum=1,
    )

    config["models"] = normalize_model_entries(config.get("models", []))
    if config["models"]:
        print("\nCurrent models:")
        for model_entry in config["models"]:
            print(f"  - {model_entry['run_name']}")
        clear_models = prompt_yes_no("Clear this model list before adding new ones?", default=False)
        if clear_models:
            config["models"] = []

    print("\nAdd models by run slug or substring. Empty input finishes selection.")
    runs = list_runs()
    if not runs:
        print("No runs with info.json found.")
        return config

    while True:
        query = input("Model slug/substring: ").strip()
        if not query:
            break

        selected_run_name = select_run_by_substring(query, runs)
        if selected_run_name is None:
            continue

        existing_run_names = {entry["run_name"] for entry in config["models"]}
        if selected_run_name in existing_run_names:
            print("Already added.")
            continue

        config["models"].append({"run_name": selected_run_name})
        print(f"Added: {selected_run_name}")

    return config


def prompt_text(label: str, current_value: str) -> str:
    raw_value = input(f"{label} [{current_value}]: ").strip()
    return raw_value or current_value


def prompt_integer(label: str, current_value: int, minimum: int) -> int:
    while True:
        raw_value = input(f"{label} [{current_value}]: ").strip()
        if not raw_value:
            return current_value

        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue

        return value


def prompt_optional_integer(label: str, current_value: Optional[int], minimum: int) -> Optional[int]:
    shown_value = "none" if current_value is None else str(current_value)
    while True:
        raw_value = input(f"{label} [{shown_value}]: ").strip().lower()
        if not raw_value:
            return current_value
        if raw_value in {"none", "all", "null"}:
            return None

        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a whole number, none, or blank.")
            continue

        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue

        return value


def prompt_choice(label: str, current_value: str, choices: set[str]) -> str:
    shown_choices = "/".join(sorted(choices))
    while True:
        raw_value = input(f"{label} ({shown_choices}) [{current_value}]: ").strip().lower()
        value = raw_value or current_value
        if value in choices:
            return value
        print(f"Please choose one of: {shown_choices}")


def prompt_yes_no(label: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw_value = input(f"{label} {suffix}: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in {"y", "yes"}:
            return True
        if raw_value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def list_runs() -> List[Tuple[str, Dict[str, Any]]]:
    runs_dir = REPOSITORY_ROOT / "runs"
    if not runs_dir.exists():
        return []

    results = []
    for info_path in sorted(runs_dir.glob("*/info.json")):
        with open(info_path, "r", encoding="utf-8") as file:
            info = json.load(file)
        results.append((info_path.parent.name, info))
    return results


def select_run_by_substring(query: str, runs: List[Tuple[str, Dict[str, Any]]]) -> Optional[str]:
    query_lower = query.lower()
    matches = [
        (run_name, run_info)
        for run_name, run_info in runs
        if query_lower in run_name.lower()
    ]

    if not matches:
        print("No matching runs.")
        return None

    if len(matches) == 1:
        return matches[0][0]

    print("\nMatches:")
    for index, (run_name, run_info) in enumerate(matches, start=1):
        status = run_info.get("status", "unknown")
        domain = run_info.get("domain", "?")
        print(f"  [{index}] {run_name} ({domain}, {status})")

    while True:
        raw_selection = input("Choose number, or blank to cancel: ").strip()
        if not raw_selection:
            return None
        try:
            selection_index = int(raw_selection) - 1
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= selection_index < len(matches):
            return matches[selection_index][0]

        print("Selection out of range.")


def validate_config(config: Dict[str, Any]) -> None:
    if not isinstance(config.get("top_k"), int) or config["top_k"] < 1:
        raise ValueError("top_k must be a positive integer.")

    if config.get("split") not in VALID_SPLITS:
        raise ValueError(f"split must be one of: {', '.join(sorted(VALID_SPLITS))}")

    if not config.get("models"):
        raise ValueError("No models selected. Add at least one run_name to config['models'].")


def run_analysis(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    top_k = int(config["top_k"])
    results = []

    for model_entry in normalize_model_entries(config["models"]):
        run_name = model_entry["run_name"]
        checkpoint_selection = model_entry.get("checkpoint", config.get("checkpoint", "latest"))
        print(f"\n=== Evaluating {run_name} ({checkpoint_selection}) ===")

        run_info = load_run_info(run_name)
        checkpoint_label, checkpoint_path = resolve_checkpoint(run_name, checkpoint_selection)
        result = evaluate_run(
            run_name=run_name,
            run_info=run_info,
            checkpoint_label=checkpoint_label,
            checkpoint_path=checkpoint_path,
            top_k=top_k,
            split=config["split"],
            max_batches=config.get("max_batches"),
            max_examples=config.get("max_examples"),
        )
        results.append(result)

    return results


def load_run_info(run_name: str) -> Dict[str, Any]:
    info_path = REPOSITORY_ROOT / "runs" / run_name / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Run info not found: {info_path}")

    with open(info_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_checkpoint(run_name: str, checkpoint_selection: str) -> Tuple[str, Path]:
    checkpoint_dir = REPOSITORY_ROOT / "runs" / run_name / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    selection = str(checkpoint_selection).strip().lower()
    epoch_checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )

    if selection == "latest":
        if epoch_checkpoints:
            latest_path = epoch_checkpoints[-1]
            return latest_path.stem.replace("checkpoint_", ""), latest_path

        temp_path = checkpoint_dir / "temp_checkpoint.pt"
        if temp_path.exists():
            return "temp", temp_path

        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if selection == "temp":
        temp_path = checkpoint_dir / "temp_checkpoint.pt"
        if temp_path.exists():
            return "temp", temp_path
        raise FileNotFoundError(f"Temp checkpoint not found: {temp_path}")

    if selection.isdigit():
        epoch_path = checkpoint_dir / f"checkpoint_epoch_{selection}.pt"
        if epoch_path.exists():
            return f"epoch_{selection}", epoch_path
        raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_path}")

    direct_path = Path(checkpoint_selection)
    if direct_path.exists():
        return direct_path.stem, direct_path

    named_path = checkpoint_dir / checkpoint_selection
    if named_path.exists():
        return named_path.stem, named_path

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_selection}' not found for run '{run_name}'."
    )


def evaluate_run(
    run_name: str,
    run_info: Dict[str, Any],
    checkpoint_label: str,
    checkpoint_path: Path,
    top_k: int,
    split: str,
    max_batches: Optional[int],
    max_examples: Optional[int],
) -> Dict[str, Any]:
    device = get_device()
    input_vocabulary = load_input_vocabulary(run_info["domain"], run_info["teacher_model"])

    model = Transformer(
        domain=run_info["domain"],
        teacher_model=run_info["teacher_model"],
        hidden_dim=run_info["hidden_dim"],
        num_layers=run_info["num_layers"],
        num_heads=run_info["num_heads"],
        dropout=run_info.get("dropout", 0.15),
        auxiliary_token_percentage=run_info.get("auxiliary_token_percentage", 1.0),
        input_vocabulary=input_vocabulary,
    ).to(device)
    load_checkpoint_into_model(model, checkpoint_path, device)
    model.eval()

    vocabulary = model.vocabulary
    output_token_to_index = {
        token: index
        for index, token in enumerate(vocabulary["token_list"])
    }
    effective_top_k = min(top_k, vocabulary["vocab_size"])
    batch_indices = get_batch_indices(run_info, split, max_batches)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus split: {split}, batches: {format_batch_range(batch_indices)}")
    print(f"top_k: {effective_top_k}")

    selected_batch_count = len(batch_indices)
    totals = MetricTotals()

    with torch.no_grad():
        for batch_index in batch_indices:
            if max_examples is not None and totals.total_examples >= max_examples:
                break

            print(f"Loading batch {batch_index + 1}...")
            totals.total_batches += 1
            for example in iter_batch_examples(run_info, batch_index):
                if max_examples is not None and totals.total_examples >= max_examples:
                    break

                example_totals = evaluate_example(
                    model=model,
                    example=example,
                    output_token_to_index=output_token_to_index,
                    top_k=effective_top_k,
                    device=device,
                )
                add_metric_totals(totals, example_totals)
                totals.total_examples += 1

                if totals.total_examples % 10 == 0:
                    metrics = totals.as_metrics()
                    print(
                        f"  {totals.total_examples} examples, "
                        f"{totals.total_steps} steps, "
                        f"top-k={metrics['top_k_accuracy']:.4f}, "
                        f"overlap={metrics['teacher_student_top_k_overlap']:.4f}, "
                        f"rank={metrics['mean_target_rank']:.2f}"
                    )

    metrics = totals.as_metrics()
    result = {
        "run_name": run_name,
        "checkpoint": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "domain": run_info["domain"],
        "teacher_model": run_info["teacher_model"],
        "top_k": effective_top_k,
        "split": split,
        "selected_batches": selected_batch_count,
        "total_batches": totals.total_batches,
        "total_examples": totals.total_examples,
        "total_steps": totals.total_steps,
        **metrics,
    }

    print(
        f"Done: top-k={result['top_k_accuracy']:.4f}, "
        f"overlap={result['teacher_student_top_k_overlap']:.4f}, "
        f"mean target rank={result['mean_target_rank']:.2f}"
    )
    return result


def load_checkpoint_into_model(model: Transformer, checkpoint_path: Path, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state = dict(checkpoint["model_state_dict"])
    model_state.pop("rotary_embedding.cos_cached", None)
    model_state.pop("rotary_embedding.sin_cached", None)
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

    if missing_keys:
        print(f"Missing checkpoint keys ignored: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected checkpoint keys ignored: {unexpected_keys}")


def get_batch_indices(
    run_info: Dict[str, Any],
    split: str,
    max_batches: Optional[int],
) -> List[int]:
    batches_dir = get_batches_dir(run_info["domain"], run_info["teacher_model"])
    batch_handler = BatchHandler(
        batches_dir=batches_dir,
        training_test_ratio=run_info.get("training_test_ratio", 0.98),
        max_training_examples=run_info.get("max_training_examples"),
        batch_size=run_info.get("batch_size", 32),
    )

    if split == "train":
        start_index, end_index = batch_handler.get_training_batches_radius()
    elif split == "test":
        start_index, end_index = batch_handler.get_test_batches_radius()
    elif split == "all":
        start_index = 0
        end_index = batch_handler._effective_batch_count()
    else:
        raise ValueError(f"Unknown split: {split}")

    if max_batches is not None:
        end_index = min(end_index, start_index + max_batches)

    return list(range(start_index, end_index))


def iter_batch_examples(run_info: Dict[str, Any], batch_index: int) -> Iterable[Dict[str, Any]]:
    batches_dir = Path(get_batches_dir(run_info["domain"], run_info["teacher_model"]))
    batch_path = batches_dir / f"batch_{batch_index + 1}.jsonl"
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    with open(batch_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                yield json.loads(stripped_line)


def evaluate_example(
    model: Transformer,
    example: Dict[str, Any],
    output_token_to_index: Dict[str, int],
    top_k: int,
    device: str,
) -> MetricTotals:
    sentence_tokens = model.tokenizer.encode(
        example["sentence"] + PROMPT_DELIMITER,
        add_special_tokens=False,
    )
    steps = example.get("steps", [])
    if not steps:
        return MetricTotals()

    token_ids = []
    teacher_logits = []
    target_indices = []
    vocab_size = model.vocabulary["vocab_size"]

    for step in steps:
        token = step["token"]
        target_index = output_token_to_index.get(token, step.get("predicted_token_index"))
        if target_index is None or target_index >= vocab_size:
            raise ValueError(f"Invalid target index for token: {token!r}")

        token_ids_for_step = model.tokenizer.encode(token, add_special_tokens=False)
        token_id = token_ids_for_step[0] if token_ids_for_step else model.tokenizer.unk_token_id
        logit_vector = step["logits"][:vocab_size]
        if len(logit_vector) < vocab_size:
            raise ValueError(
                f"Teacher logit vector is shorter than the student vocabulary: "
                f"{len(logit_vector)} < {vocab_size}"
            )

        token_ids.append(token_id)
        teacher_logits.append(logit_vector)
        target_indices.append(target_index)

    if not target_indices:
        return MetricTotals()

    remapped_input_ids = model.remap_input_tokens(sentence_tokens + token_ids[:-1])
    input_tensor = torch.tensor([remapped_input_ids], dtype=torch.long, device=device)
    teacher_logits_tensor = torch.tensor(teacher_logits, dtype=torch.float32, device=device)
    target_indices_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)

    model_logits = model(input_tensor)
    sentence_length = len(sentence_tokens)
    prediction_logits = model_logits[
        0,
        sentence_length - 1:sentence_length - 1 + len(target_indices),
        :,
    ]

    student_top_indices = torch.topk(prediction_logits, k=top_k, dim=-1).indices
    teacher_top_indices = torch.topk(teacher_logits_tensor, k=top_k, dim=-1).indices

    top_k_hits = (
        student_top_indices == target_indices_tensor.unsqueeze(1)
    ).any(dim=1).sum().item()

    top_k_overlap_counts = (
        student_top_indices.unsqueeze(2) == teacher_top_indices.unsqueeze(1)
    ).any(dim=2).sum(dim=1)
    top_k_overlap_sum = (top_k_overlap_counts.float() / top_k).sum().item()

    target_logits = prediction_logits.gather(1, target_indices_tensor.unsqueeze(1))
    target_ranks = (prediction_logits > target_logits).sum(dim=1) + 1
    target_rank_sum = target_ranks.float().sum().item()

    return MetricTotals(
        top_k_hits=int(top_k_hits),
        top_k_overlap_sum=float(top_k_overlap_sum),
        target_rank_sum=float(target_rank_sum),
        total_steps=len(target_indices),
    )


def add_metric_totals(total: MetricTotals, addition: MetricTotals) -> None:
    total.top_k_hits += addition.top_k_hits
    total.top_k_overlap_sum += addition.top_k_overlap_sum
    total.target_rank_sum += addition.target_rank_sum
    total.total_steps += addition.total_steps


def format_batch_range(batch_indices: List[int]) -> str:
    if not batch_indices:
        return "none"
    return f"{batch_indices[0] + 1}-{batch_indices[-1] + 1} ({len(batch_indices)} batches)"


def write_outputs(config: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "results": results,
    }

    json_path = output_dir / "top_k_results.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    csv_path = output_dir / "top_k_results.csv"
    write_csv(csv_path, results)
    write_plots(output_dir, results)

    print("\nSaved outputs:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'top_k_accuracy.png'}")
    print(f"  {output_dir / 'teacher_student_top_k_overlap.png'}")
    print(f"  {output_dir / 'mean_target_rank.png'}")


def write_csv(csv_path: Path, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "checkpoint",
        "domain",
        "teacher_model",
        "top_k",
        "split",
        "selected_batches",
        "total_batches",
        "total_examples",
        "total_steps",
        "top_k_accuracy",
        "teacher_student_top_k_overlap",
        "mean_target_rank",
        "checkpoint_path",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fieldnames})


def write_plots(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so plots were not generated.")
        return

    plot_specs = [
        (
            "top_k_accuracy",
            "Top-k Accuracy",
            "Accuracy",
            output_dir / "top_k_accuracy.png",
            True,
        ),
        (
            "teacher_student_top_k_overlap",
            "Teacher-Student Top-k Overlap",
            "Overlap",
            output_dir / "teacher_student_top_k_overlap.png",
            True,
        ),
        (
            "mean_target_rank",
            "Mean Target Rank",
            "Rank (lower is better)",
            output_dir / "mean_target_rank.png",
            False,
        ),
    ]

    for metric_key, title, y_label, path, clamp_to_unit in plot_specs:
        save_bar_plot(plt, results, metric_key, title, y_label, path, clamp_to_unit)


def save_bar_plot(
    plt: Any,
    results: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    y_label: str,
    path: Path,
    clamp_to_unit: bool,
) -> None:
    labels = [make_plot_label(result) for result in results]
    values = [result[metric_key] for result in results]
    width = max(8, min(24, len(results) * 2.2))

    plt.figure(figsize=(width, 5))
    bars = plt.bar(range(len(results)), values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(range(len(results)), labels, rotation=25, ha="right")
    if clamp_to_unit:
        plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if clamp_to_unit else f"{value:.1f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_plot_label(result: Dict[str, Any]) -> str:
    run_name = result["run_name"]
    checkpoint = result["checkpoint"]
    if checkpoint and checkpoint not in run_name:
        return f"{run_name}\n{checkpoint}"
    return run_name


if __name__ == "__main__":
    main()
