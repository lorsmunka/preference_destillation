"""Evaluation over test batches: teacher-forced pass (loss, token accuracy, distribution
metrics) plus a free-running rollout per example (student accuracy, task metric,
natural termination). Used for the per-epoch eval and the mini-evals during training."""

import math
from time import time
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from library.shared.metrics.distribution import step_distribution_stats
from library.training.example_encoder import ExampleEncoder


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

    def evaluate_batches(self, batch_start: int, batch_end: int, verbose: bool = False) -> Dict:
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
            for batch_idx in range(batch_start, batch_end):
                batch_start_time = time()
                batch_data = self.batch_handler.get_batch(batch_idx)

                batch_loss = 0.0
                batch_kl_loss = 0.0
                batch_ce_loss = 0.0
                batch_teacher_forced_correct = 0
                batch_student_correct = 0
                batch_steps = 0

                for example in batch_data:
                    (loss_sum, kl_loss_sum, ce_loss_sum, teacher_forced_correct, student_correct,
                     num_steps, student_tokens, distribution, terminated, generated_length) = self.eval_single_example(example)
                    batch_loss += loss_sum
                    batch_kl_loss += kl_loss_sum
                    batch_ce_loss += ce_loss_sum
                    batch_teacher_forced_correct += teacher_forced_correct
                    batch_student_correct += student_correct
                    batch_steps += num_steps
                    total_loss += loss_sum
                    total_kl_loss += kl_loss_sum
                    total_ce_loss += ce_loss_sum
                    total_teacher_forced_correct += teacher_forced_correct
                    total_student_correct += student_correct
                    total_steps += num_steps

                    for key in distribution_totals:
                        distribution_totals[key] += distribution.get(key, 0)
                    terminated_count += 1 if terminated else 0
                    length_ratio_sum += (generated_length / num_steps) if num_steps > 0 else 0.0
                    eval_examples += 1

                    ground_truth_response = example.get('model_response', '')
                    task_accuracy_calculator.update(student_tokens, ground_truth_response)

                if verbose:
                    batch_elapsed = time() - batch_start_time
                    avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
                    avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
                    avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
                    batch_tf_accuracy = batch_teacher_forced_correct / batch_steps if batch_steps > 0 else 0.0
                    batch_student_accuracy = batch_student_correct / batch_steps if batch_steps > 0 else 0.0
                    running_task_accuracy = task_accuracy_calculator.result().accuracy
                    print(f"Eval Batch {batch_idx + 1}: {batch_steps} steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, tf_acc={batch_tf_accuracy:.4f}, student_acc={batch_student_accuracy:.4f}, task_acc={running_task_accuracy:.4f} -> took {batch_elapsed:.2f}s")

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

        return {
            'avg_loss': avg_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_ce_loss': avg_ce_loss,
            'teacher_forced_accuracy': teacher_forced_accuracy,
            'student_accuracy': student_accuracy,
            'task_accuracy': metric_result.accuracy,
            'confusion_matrices': metric_result.confusion or {},
            'distribution': distribution,
            'total_steps': total_steps,
        }

    def eval_single_example(self, example: Dict) -> Tuple:
        if len(example['steps']) == 0:
            return 0.0, 0.0, 0.0, 0, 0, 0, [], {}, False, 0

        encoded = self.encoder.encode(example)
        num_steps = encoded.num_steps

        # Teacher-forced: single forward pass with all ground-truth tokens as input.
        model_logits = self.model(encoded.input_tensor)
        sentence_length = len(encoded.sentence_tokens)
        prediction_logits = model_logits[0, sentence_length - 1:sentence_length - 1 + num_steps, :]

        kl_loss, ce_loss, total_loss = self.loss_function(
            prediction_logits, encoded.target_logits_tensor, encoded.target_indices_tensor, 'sum')

        teacher_forced_correct = (torch.argmax(prediction_logits, dim=-1) == encoded.target_indices_tensor).sum().item()

        # Distribution metrics — free, from the teacher-forced pass already computed.
        distribution = step_distribution_stats(
            prediction_logits, encoded.target_logits_tensor, encoded.target_indices_tensor, self.eval_top_k)
        distribution['ce_sum'] = F.cross_entropy(
            prediction_logits, encoded.target_indices_tensor, reduction='sum').item()

        # Student: free-generation rollout (own predictions), capped at eval_cap_multiple x
        # teacher length. Stops on the domain stop token -> natural-termination signal.
        student_correct = 0
        remapped_sentence_tokens = self.model.remap_input_tokens(encoded.sentence_tokens)
        student_token_ids = []
        student_tokens = []
        generated_text = ""
        terminated = False
        max_rollout = max(num_steps, int(self.eval_cap_multiple * num_steps))

        for step_index in range(max_rollout):
            student_input_tensor = torch.tensor(
                [remapped_sentence_tokens + student_token_ids], dtype=torch.long, device=self.device)
            student_logits = self.model(student_input_tensor)[:, -1, :]
            student_predicted_index = torch.argmax(student_logits[0]).item()
            student_predicted_token_id = self.model.output_token_ids[student_predicted_index]
            student_predicted_token = self.token_list[student_predicted_index]
            if step_index < num_steps and student_predicted_index == encoded.target_indices[step_index]:
                student_correct += 1
            student_token_ids.append(self.model.remap_input_tokens([student_predicted_token_id])[0])
            student_tokens.append(student_predicted_token)
            decoded = self.tokenizer.decode([student_predicted_token_id])
            generated_text += decoded
            if self.domain.is_stop(decoded, generated_text):
                terminated = True
                break

        return (total_loss.item(), kl_loss.item(), ce_loss.item(), teacher_forced_correct,
                student_correct, num_steps, student_tokens, distribution, terminated, len(student_tokens))
