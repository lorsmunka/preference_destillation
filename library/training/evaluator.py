"""Evaluation over test batches: teacher-forced pass (loss, token accuracy, distribution
metrics) plus a free-running rollout per example (student accuracy, task metric,
natural termination). Used for the per-epoch eval and the mini-evals during training.

The rollouts of one batch file run together with a key/value cache (IncrementalDecoder):
one forward pass over one new position per token for the whole file, instead of one pass
over the whole growing prefix per token per example. Every example still decodes greedily
on its own, stops on the domain stop rule or at eval_cap_multiple x its teacher length, and
is scored exactly as before.
"""

import math
from dataclasses import dataclass, field
from time import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from library.model.transformer import IncrementalDecoder
from library.shared.metrics.distribution import step_distribution_stats
from library.training.example_encoder import EncodedExample, ExampleEncoder


@dataclass
class Rollout:
    tokens: List[str] = field(default_factory=list)   # output-vocabulary token strings
    text: str = ""                                     # the decoded tokens, joined
    correct: int = 0                                   # positions matching the teacher's token
    terminated: bool = False                           # stopped on the domain stop rule


def majority_baseline(confusion_matrices: Dict[str, Dict[str, Dict[str, int]]]) -> Optional[float]:
    """Accuracy of always predicting each field's most common true label: per field the
    largest confusion-matrix row sum, pooled over fields the way task accuracy pools them
    (correct field values / scored field values). None when there are no matrices."""
    majority_count = 0
    scored_count = 0
    for matrix in confusion_matrices.values():
        row_sums = [sum(row.values()) for row in matrix.values()]
        if row_sums:
            majority_count += max(row_sums)
            scored_count += sum(row_sums)
    return majority_count / scored_count if scored_count else None


class Evaluator:
    def __init__(self, model, domain, encoder: ExampleEncoder, batch_handler, device,
                 loss_function: Callable, eval_top_k: int, eval_cap_multiple: float):
        self.model = model
        self.domain = domain
        self.encoder = encoder
        self.batch_handler = batch_handler
        self.device = device
        self.loss_function = loss_function   # (prediction_logits, target_logits, target_indices, reduction) -> (kl, ce, total)
        self.eval_top_k = eval_top_k
        self.eval_cap_multiple = eval_cap_multiple
        self.tokenizer = model.tokenizer
        self.token_list = model.vocabulary['token_list']

    def evaluate_batches(self, batch_count: Optional[int] = None, verbose: bool = False) -> Dict:
        """Evaluates the first `batch_count` test batches (None = all of them)."""
        if batch_count is None:
            batch_count = self.batch_handler.test_batch_count()
        task_accuracy_calculator = self.domain.task_metric()

        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_teacher_forced_correct = 0
        total_student_correct = 0
        total_steps = 0

        distribution_totals = {'topk_hits': 0.0, 'overlap_sum': 0.0, 'target_rank_sum': 0.0,
                               'student_entropy_sum': 0.0, 'teacher_entropy_sum': 0.0,
                               'ce_sum': 0.0, 'steps': 0}
        terminated_count = 0
        length_ratio_sum = 0.0
        eval_examples = 0

        with torch.no_grad():
            for batch_index in range(batch_count):
                batch_start_time = time()
                batch_data = self.batch_handler.get_test_batch(batch_index)
                encoded_examples = [self.encoder.encode(example) if example['steps'] else None
                                    for example in batch_data]
                rollouts = iter(self.rollouts([encoded for encoded in encoded_examples if encoded is not None]))

                batch_loss = 0.0
                batch_kl_loss = 0.0
                batch_ce_loss = 0.0
                batch_teacher_forced_correct = 0
                batch_student_correct = 0
                batch_steps = 0

                for example, encoded in zip(batch_data, encoded_examples):
                    if encoded is None:   # no teacher steps: nothing to score, but it still counts as an example
                        loss_sum, kl_loss_sum, ce_loss_sum, teacher_forced_correct, distribution = 0.0, 0.0, 0.0, 0, {}
                        num_steps, rollout = 0, Rollout()
                    else:
                        loss_sum, kl_loss_sum, ce_loss_sum, teacher_forced_correct, distribution = self.teacher_forced(encoded)
                        num_steps, rollout = encoded.num_steps, next(rollouts)
                    batch_loss += loss_sum
                    batch_kl_loss += kl_loss_sum
                    batch_ce_loss += ce_loss_sum
                    batch_teacher_forced_correct += teacher_forced_correct
                    batch_student_correct += rollout.correct
                    batch_steps += num_steps
                    total_loss += loss_sum
                    total_kl_loss += kl_loss_sum
                    total_ce_loss += ce_loss_sum
                    total_teacher_forced_correct += teacher_forced_correct
                    total_student_correct += rollout.correct
                    total_steps += num_steps

                    for key in distribution_totals:
                        distribution_totals[key] += distribution.get(key, 0)
                    terminated_count += 1 if rollout.terminated else 0
                    length_ratio_sum += (len(rollout.tokens) / num_steps) if num_steps > 0 else 0.0
                    eval_examples += 1

                    ground_truth_response = example.get('model_response', '')
                    task_accuracy_calculator.update(rollout.tokens, ground_truth_response)

                if verbose:
                    batch_elapsed = time() - batch_start_time
                    avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
                    avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
                    avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
                    batch_tf_accuracy = batch_teacher_forced_correct / batch_steps if batch_steps > 0 else 0.0
                    batch_student_accuracy = batch_student_correct / batch_steps if batch_steps > 0 else 0.0
                    running_task_accuracy = task_accuracy_calculator.result().accuracy
                    print(f"Eval Batch {batch_index + 1}: {batch_steps} steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, tf_acc={batch_tf_accuracy:.4f}, student_acc={batch_student_accuracy:.4f}, task_acc={running_task_accuracy:.4f} -> took {batch_elapsed:.2f}s")

        steps = total_steps if total_steps > 0 else 1
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_kl_loss = total_kl_loss / total_steps if total_steps > 0 else 0.0
        avg_ce_loss = total_ce_loss / total_steps if total_steps > 0 else 0.0
        teacher_forced_accuracy = total_teacher_forced_correct / total_steps if total_steps > 0 else 0.0
        student_accuracy = total_student_correct / total_steps if total_steps > 0 else 0.0
        metric_result = task_accuracy_calculator.result()

        distribution = {
            'topk_accuracy': distribution_totals['topk_hits'] / steps,
            'teacher_student_overlap': distribution_totals['overlap_sum'] / steps,
            'mean_target_rank': distribution_totals['target_rank_sum'] / steps,
            'student_entropy': distribution_totals['student_entropy_sum'] / steps,
            'teacher_entropy': distribution_totals['teacher_entropy_sum'] / steps,
            'perplexity': math.exp(distribution_totals['ce_sum'] / steps),
            'validity_rate': metric_result.validity_rate,
            'termination_rate': terminated_count / eval_examples if eval_examples > 0 else 0.0,
            'mean_length_ratio': length_ratio_sum / eval_examples if eval_examples > 0 else 0.0,
        }

        # The floor task accuracy must clear to mean anything (code review X-C1): logged
        # beside it whenever the task metric has confusion matrices (closed-label fields).
        confusion_matrices = metric_result.confusion or {}
        baseline = majority_baseline(confusion_matrices)
        baseline_fields = {}
        if baseline is not None:
            baseline_fields = {'majority_baseline': round(baseline, 4),
                               'task_accuracy_excess': round(metric_result.accuracy - baseline, 4)}

        return {
            'avg_loss': avg_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_ce_loss': avg_ce_loss,
            'teacher_forced_accuracy': teacher_forced_accuracy,
            'student_accuracy': student_accuracy,
            'task_accuracy': metric_result.accuracy,
            'confusion_matrices': confusion_matrices,
            'baseline': baseline_fields,
            'distribution': distribution,
            'total_steps': total_steps,
        }

    def teacher_forced(self, encoded: EncodedExample):
        """Single forward pass with every ground-truth token as input: loss sums, token
        accuracy and the distribution metrics."""
        num_steps = encoded.num_steps
        model_logits = self.model(encoded.input_tensor)
        sentence_length = len(encoded.sentence_tokens)
        prediction_logits = model_logits[0, sentence_length - 1:sentence_length - 1 + num_steps, :]

        kl_loss, ce_loss, total_loss = self.loss_function(
            prediction_logits, encoded.target_logits_tensor, encoded.target_indices_tensor, 'sum')

        teacher_forced_correct = (torch.argmax(prediction_logits, dim=-1) == encoded.target_indices_tensor).sum().item()

        distribution = step_distribution_stats(
            prediction_logits, encoded.target_logits_tensor, encoded.target_indices_tensor, self.eval_top_k)
        distribution['ce_sum'] = F.cross_entropy(
            prediction_logits, encoded.target_indices_tensor, reduction='sum').item()

        return total_loss.item(), kl_loss.item(), ce_loss.item(), teacher_forced_correct, distribution

    def rollouts(self, encoded_examples: List[EncodedExample]) -> List[Rollout]:
        """Greedy free generation (own predictions as context) for every example at once.
        Each is capped at eval_cap_multiple x its teacher length and stops on the domain stop
        rule, the natural-termination signal."""
        if not encoded_examples:
            return []
        prompts = [self.model.remap_input_tokens(encoded.sentence_tokens) for encoded in encoded_examples]
        caps = [max(encoded.num_steps, int(self.eval_cap_multiple * encoded.num_steps)) for encoded in encoded_examples]
        results = [Rollout() for _ in encoded_examples]

        decoder = IncrementalDecoder(self.model, prompts)
        running = list(range(len(encoded_examples)))   # example index of each decoder row
        while running:
            predicted_indices = decoder.next_logits().argmax(dim=-1).tolist()
            continuing_rows, next_input_ids = [], []
            for row, example_index in enumerate(running):
                predicted_index = predicted_indices[row]
                encoded, rollout = encoded_examples[example_index], results[example_index]
                step_index = len(rollout.tokens)
                if step_index < encoded.num_steps and predicted_index == encoded.target_indices[step_index]:
                    rollout.correct += 1
                predicted_token_id = self.model.output_token_ids[predicted_index]
                rollout.tokens.append(self.token_list[predicted_index])
                decoded = self.tokenizer.decode([predicted_token_id])
                rollout.text += decoded
                if self.domain.is_stop(decoded, rollout.text):
                    rollout.terminated = True
                elif len(rollout.tokens) < caps[example_index]:
                    continuing_rows.append(row)
                    next_input_ids.append(self.model.remap_input_tokens([predicted_token_id])[0])
            if not continuing_rows:
                break
            if len(continuing_rows) < len(running):
                decoder.keep_rows(continuing_rows)
                running = [running[row] for row in continuing_rows]
            decoder.append(next_input_ids)
        return results
