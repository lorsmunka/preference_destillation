import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import LOGS_DIR


class TrainingAnalyzer:
    def __init__(self, data: List[Dict]):
        self.train_batches = [d for d in data if d.get('type') == 'train_batch']
        self.train_epochs = [d for d in data if d.get('type') == 'train_epoch']
        self.eval_epochs = [d for d in data if d.get('type') == 'eval_epoch']

    def print_summary(self):
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)

        if not self.train_batches:
            print("No training data found.")
            return

        sessions = set(b.get('session_id', 'unknown') for b in self.train_batches)
        total_steps = sum(b['steps'] for b in self.train_batches)
        total_time = sum(b['time_seconds'] for b in self.train_batches)

        print(f"\nBatches completed: {len(self.train_batches):,}")
        print(f"Epochs completed: {len(self.train_epochs)}")
        print(f"Sessions: {len(sessions)}")
        print(f"Total steps: {total_steps:,}")
        print(f"Total time: {total_time / 3600:.2f} hours")

        losses = [b['loss'] for b in self.train_batches]
        accuracies = [b['accuracy'] for b in self.train_batches]

        print(f"\nLoss - First: {losses[0]:.4f}, Last: {losses[-1]:.4f}")
        print(f"Loss - Min: {min(losses):.4f}, Max: {max(losses):.4f}")
        print(f"Loss - Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

        print(f"\nAccuracy - First: {accuracies[0] * 100:.1f}%, Last: {accuracies[-1] * 100:.1f}%")
        print(f"Accuracy - Min: {min(accuracies) * 100:.1f}%, Max: {max(accuracies) * 100:.1f}%")

        if self.eval_epochs:
            latest_eval = self.eval_epochs[-1]
            print(f"\nLatest evaluation (epoch {latest_eval['epoch']}):")
            print(f"  Loss: {latest_eval['avg_loss']:.4f}")
            teacher_forced_acc = latest_eval.get('teacher_forced_accuracy', latest_eval.get('accuracy', 0))
            student_acc = latest_eval.get('student_accuracy', 0)
            classification_acc = latest_eval.get('classification_accuracy', 0)
            print(f"  Teacher-Forced Accuracy: {teacher_forced_acc * 100:.1f}%")
            print(f"  Student Accuracy: {student_acc * 100:.1f}%")
            print(f"  Classification Accuracy: {classification_acc * 100:.1f}%")

            confusion = latest_eval.get('confusion_matrices')
            if confusion:
                print(f"\n  Confusion Matrices:")
                for category, matrix in confusion.items():
                    labels = list(matrix.keys())
                    print(f"\n    {category.upper()}")
                    header = "    " + f"{'':>12s}" + "".join(f"{l:>12s}" for l in labels)
                    print(header)
                    for true_label in labels:
                        row_counts = [str(matrix[true_label].get(pred, 0)) for pred in labels]
                        print(f"    {true_label:>12s}" + "".join(f"{c:>12s}" for c in row_counts))

        if len(self.train_batches) >= 100:
            recent = self.train_batches[-100:]
            early = self.train_batches[:100]
            recent_avg_loss = np.mean([b['loss'] for b in recent])
            early_avg_loss = np.mean([b['loss'] for b in early])
            print(f"\nTrend (first 100 vs last 100 batches):")
            print(f"  Avg loss: {early_avg_loss:.4f} -> {recent_avg_loss:.4f}")
            print(f"  Change: {((early_avg_loss - recent_avg_loss) / early_avg_loss * 100):.1f}%")

        lrs = [b['learning_rate'] for b in self.train_batches]
        print(f"\nLearning rate - Start: {lrs[0]:.2e}, Current: {lrs[-1]:.2e}")

        kl_ratios = [b.get('kl_ratio', 0) for b in self.train_batches]
        if any(kl_ratios):
            print(f"KL ratio - Start: {kl_ratios[0]:.4f}, Current: {kl_ratios[-1]:.4f}")

    def plot(self, moving_average_func, exponential_moving_average_func, show=True):
        if not self.train_batches:
            print("No training data to plot.")
            return

        fig = plt.figure(figsize=(16, 62))
        gs = GridSpec(11, 2, figure=fig, hspace=0.35, wspace=0.25, height_ratios=[0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        losses = [b['loss'] for b in self.train_batches]
        kl_losses = [b.get('kl_loss', 0) for b in self.train_batches]
        ce_losses = [b.get('ce_loss', 0) for b in self.train_batches]
        accuracies = [b['accuracy'] * 100 for b in self.train_batches]
        lrs = [b['learning_rate'] for b in self.train_batches]
        batch_indices = list(range(1, len(self.train_batches) + 1))

        # Calculate summary stats (using 200 batch windows)
        current_acc = np.mean(accuracies[-200:]) if len(accuracies) >= 200 else accuracies[-1] if accuracies else 0
        previous_acc = np.mean(accuracies[-400:-200]) if len(accuracies) >= 400 else (accuracies[0] if accuracies else 0)
        acc_diff = current_acc - previous_acc
        acc_trend = f"↑ +{acc_diff:.1f}%" if acc_diff > 0 else f"↓ {acc_diff:.1f}%" if acc_diff < 0 else "→"

        current_loss = np.mean(losses[-200:]) if len(losses) >= 200 else losses[-1] if losses else 0
        previous_loss = np.mean(losses[-400:-200]) if len(losses) >= 400 else (losses[0] if losses else 0)
        loss_diff = current_loss - previous_loss
        loss_trend = f"↓ {loss_diff:.4f}" if loss_diff < 0 else f"↑ +{loss_diff:.4f}" if loss_diff > 0 else "→"

        current_ce = np.mean(ce_losses[-200:]) if len(ce_losses) >= 200 else ce_losses[-1] if ce_losses else 0
        current_kl = np.mean(kl_losses[-200:]) if len(kl_losses) >= 200 else kl_losses[-1] if kl_losses else 0

        epochs_done = len(self.train_epochs)

        # Summary panel
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')

        summary_text = (
            f"Accuracy: {current_acc:.1f}% ({acc_trend})        "
            f"Loss: {current_loss:.4f} ({loss_trend})        "
            f"CE: {current_ce:.4f}        "
            f"KL: {current_kl:.4f}        "
            f"Batches: {len(self.train_batches):,}        "
            f"Epochs: {epochs_done}"
        )
        ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=14, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

        # Prepare recent 1000 window data (left side)
        window_1000 = min(1000, len(losses))
        if len(losses) > window_1000:
            losses_1000 = losses[-window_1000:]
            kl_losses_1000 = kl_losses[-window_1000:]
            ce_losses_1000 = ce_losses[-window_1000:]
            accuracies_1000 = accuracies[-window_1000:]
            indices_1000 = batch_indices[-window_1000:]
        else:
            losses_1000 = losses
            kl_losses_1000 = kl_losses
            ce_losses_1000 = ce_losses
            accuracies_1000 = accuracies
            indices_1000 = batch_indices

        # All data (right side) - y-axis will be constrained based on last 90% of data
        losses_all = losses
        kl_losses_all = kl_losses
        ce_losses_all = ce_losses
        accuracies_all = accuracies
        indices_all = batch_indices

        # Calculate start index for last 90% of data (ignore first 10%)
        last_90_start = int(len(losses) * 0.1)
        accuracies_last_90 = accuracies[last_90_start:] if last_90_start < len(accuracies) else accuracies
        losses_last_90 = losses[last_90_start:] if last_90_start < len(losses) else losses
        ce_losses_last_90 = ce_losses[last_90_start:] if last_90_start < len(ce_losses) else ce_losses
        kl_losses_last_90 = kl_losses[last_90_start:] if last_90_start < len(kl_losses) else kl_losses

        # Accuracy - All (y-axis based on last 90% of data)
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(indices_all, accuracies_all, color='lightgreen', alpha=0.5, linewidth=0.7)
        ma_window_all = max(40, int(len(accuracies_all) * 0.03))
        if len(accuracies_all) >= ma_window_all:
            ma_acc = moving_average_func(accuracies_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_acc, color='green', linewidth=2, label=f'MA({ma_window_all})')
        if len(accuracies_all) >= 200:
            last_200_avg = np.mean(accuracies_all[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.1f}%')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy - All ({len(accuracies_all)} batches, y-axis: last 90%)')
        min_acc_90 = min(accuracies_last_90)
        max_acc_90 = max(accuracies_last_90)
        padding = (max_acc_90 - min_acc_90) * 0.1 if max_acc_90 > min_acc_90 else 5
        ax.set_ylim([max(0, min_acc_90 - padding), min(100, max_acc_90 + padding)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy - Recent 1000
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(indices_1000, accuracies_1000, color='lightgreen', alpha=0.5, linewidth=0.7)
        if len(accuracies_1000) >= 40:
            ma40_acc = moving_average_func(accuracies_1000, 40)
            ax.plot(indices_1000[39:], ma40_acc, color='green', linewidth=2, label='MA(40)')
        if len(accuracies_1000) >= 200:
            last_200_avg = np.mean(accuracies_1000[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.1f}%')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy - Recent ({window_1000} batches)')
        min_acc = min(accuracies_1000)
        max_acc = max(accuracies_1000)
        padding = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 5
        ax.set_ylim([max(0, min_acc - padding), min(100, max_acc + padding)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss - All (y-axis based on last 90% of data)
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(indices_all, losses_all, color='lightcoral', alpha=0.5, linewidth=0.7)
        if len(losses_all) >= ma_window_all:
            ma_loss = moving_average_func(losses_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_loss, color='red', linewidth=2, label=f'MA({ma_window_all})')
        if len(losses_all) >= 200:
            last_200_avg = np.mean(losses_all[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.4f}')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss - All ({len(losses_all)} batches, y-axis: last 90%)')
        min_loss_90 = min(losses_last_90)
        max_loss_90 = max(losses_last_90)
        padding = (max_loss_90 - min_loss_90) * 0.1 if max_loss_90 > min_loss_90 else 0.1
        ax.set_ylim([max(0, min_loss_90 - padding), max_loss_90 + padding])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss - Recent 1000
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(indices_1000, losses_1000, color='lightcoral', alpha=0.5, linewidth=0.7)
        if len(losses_1000) >= 40:
            ma40_loss = moving_average_func(losses_1000, 40)
            ax.plot(indices_1000[39:], ma40_loss, color='red', linewidth=2, label='MA(40)')
        if len(losses_1000) >= 200:
            last_200_avg = np.mean(losses_1000[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.4f}')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss - Recent ({window_1000} batches)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # CE Loss - All (y-axis based on last 90% of data)
        ax = fig.add_subplot(gs[3, 0])
        ax.plot(indices_all, ce_losses_all, color='lightskyblue', alpha=0.5, linewidth=0.7)
        if len(ce_losses_all) >= ma_window_all:
            ma_ce = moving_average_func(ce_losses_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_ce, color='blue', linewidth=2, label=f'MA({ma_window_all})')
        if len(ce_losses_all) >= 200:
            last_200_avg = np.mean(ce_losses_all[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.4f}')
        ax.set_xlabel('Batch')
        ax.set_ylabel('CE Loss')
        ax.set_title(f'CE Loss - All ({len(ce_losses_all)} batches, y-axis: last 90%)')
        min_ce_90 = min(ce_losses_last_90)
        max_ce_90 = max(ce_losses_last_90)
        padding = (max_ce_90 - min_ce_90) * 0.1 if max_ce_90 > min_ce_90 else 0.1
        ax.set_ylim([max(0, min_ce_90 - padding), max_ce_90 + padding])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # CE Loss - Recent 1000
        ax = fig.add_subplot(gs[3, 1])
        ax.plot(indices_1000, ce_losses_1000, color='lightskyblue', alpha=0.5, linewidth=0.7)
        if len(ce_losses_1000) >= 40:
            ma40_ce = moving_average_func(ce_losses_1000, 40)
            ax.plot(indices_1000[39:], ma40_ce, color='blue', linewidth=2, label='MA(40)')
        if len(ce_losses_1000) >= 200:
            last_200_avg = np.mean(ce_losses_1000[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.4f}')
        ax.set_xlabel('Batch')
        ax.set_ylabel('CE Loss')
        ax.set_title(f'CE Loss - Recent ({window_1000} batches)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # KL Loss - All (y-axis based on last 90% of data)
        ax = fig.add_subplot(gs[4, 0])
        ax.plot(indices_all, kl_losses_all, color='plum', alpha=0.5, linewidth=0.7)
        if len(kl_losses_all) >= ma_window_all:
            ma_kl = moving_average_func(kl_losses_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_kl, color='purple', linewidth=2, label=f'MA({ma_window_all})')
        if len(kl_losses_all) >= 200:
            last_200_avg = np.mean(kl_losses_all[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.4f}')
        ax.set_xlabel('Batch')
        ax.set_ylabel('KL Loss')
        ax.set_title(f'KL Loss - All ({len(kl_losses_all)} batches, y-axis: last 90%)')
        min_kl_90 = min(kl_losses_last_90)
        max_kl_90 = max(kl_losses_last_90)
        padding = (max_kl_90 - min_kl_90) * 0.1 if max_kl_90 > min_kl_90 else 0.1
        ax.set_ylim([max(0, min_kl_90 - padding), max_kl_90 + padding])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # KL Loss - Recent 1000
        ax = fig.add_subplot(gs[4, 1])
        ax.plot(indices_1000, kl_losses_1000, color='plum', alpha=0.5, linewidth=0.7)
        if len(kl_losses_1000) >= 40:
            ma40_kl = moving_average_func(kl_losses_1000, 40)
            ax.plot(indices_1000[39:], ma40_kl, color='purple', linewidth=2, label='MA(40)')
        if len(kl_losses_1000) >= 200:
            last_200_avg = np.mean(kl_losses_1000[-200:])
            ax.axhline(y=last_200_avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {last_200_avg:.4f}')
        ax.set_xlabel('Batch')
        ax.set_ylabel('KL Loss')
        ax.set_title(f'KL Loss - Recent ({window_1000} batches)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Epoch Summary - Loss
        ax = fig.add_subplot(gs[5, 0])
        if self.train_epochs and self.eval_epochs:
            train_ep = [d['epoch'] for d in self.train_epochs]
            train_loss = [d['avg_loss'] for d in self.train_epochs]
            eval_ep = [d['epoch'] for d in self.eval_epochs]
            eval_loss = [d['avg_loss'] for d in self.eval_epochs]
            ax.plot(train_ep, train_loss, marker='o', color='red', linewidth=2, label='Train')
            ax.plot(eval_ep, eval_loss, marker='s', color='blue', linewidth=2, label='Eval')
            ax.legend()
        elif self.train_epochs:
            train_ep = [d['epoch'] for d in self.train_epochs]
            train_loss = [d['avg_loss'] for d in self.train_epochs]
            ax.plot(train_ep, train_loss, marker='o', color='red', linewidth=2, label='Train')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No epoch data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Epoch Summary - Loss')
        ax.grid(True, alpha=0.3)

        # Epoch Summary - Eval Accuracy
        ax = fig.add_subplot(gs[5, 1])
        if self.eval_epochs:
            eval_ep = [d['epoch'] for d in self.eval_epochs]
            teacher_forced_acc = [d.get('teacher_forced_accuracy', d.get('accuracy', 0)) * 100 for d in self.eval_epochs]
            student_acc = [d.get('student_accuracy', 0) * 100 for d in self.eval_epochs]
            classification_acc = [d.get('classification_accuracy', 0) * 100 for d in self.eval_epochs]
            ax.plot(eval_ep, teacher_forced_acc, marker='o', color='green', linewidth=2, label='Teacher-Forced')
            ax.plot(eval_ep, student_acc, marker='s', color='blue', linewidth=2, label='Student')
            ax.plot(eval_ep, classification_acc, marker='^', color='orange', linewidth=2, label='Classification')
            all_acc = teacher_forced_acc + student_acc + classification_acc
            min_acc = min(all_acc)
            max_acc = max(all_acc)
            padding = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 5
            ax.set_ylim([max(0, min_acc - padding), min(100, max_acc + padding)])
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No evaluation data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Epoch Summary - Eval Accuracy')
        ax.grid(True, alpha=0.3)

        # Learning Rate Schedule
        ax = fig.add_subplot(gs[6, 0])
        ax.plot(batch_indices, lrs, color='orange', linewidth=2, label='Learning Rate')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # KL Ratio Schedule
        ax = fig.add_subplot(gs[6, 1])
        kl_ratios = [b.get('kl_ratio', 0) for b in self.train_batches]
        if any(kl_ratios):
            ax.plot(batch_indices, kl_ratios, color='purple', linewidth=2, label='KL Ratio')
            ax.set_ylim([0, 1])
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No KL ratio data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Batch')
        ax.set_ylabel('KL Ratio')
        ax.set_title('KL Ratio Schedule')
        ax.grid(True, alpha=0.3)

        # ========== FULL RANGE CHARTS (all data for y-axis limits) ==========

        # Accuracy - Full Range
        ax = fig.add_subplot(gs[7, 0])
        ax.plot(indices_all, accuracies_all, color='lightgreen', alpha=0.5, linewidth=0.7)
        if len(accuracies_all) >= ma_window_all:
            ma_acc = moving_average_func(accuracies_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_acc, color='green', linewidth=2, label=f'MA({ma_window_all})')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy - Full Range ({len(accuracies_all)} batches)')
        min_acc_full = min(accuracies_all)
        max_acc_full = max(accuracies_all)
        padding = (max_acc_full - min_acc_full) * 0.05 if max_acc_full > min_acc_full else 5
        ax.set_ylim([max(0, min_acc_full - padding), min(100, max_acc_full + padding)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss - Full Range
        ax = fig.add_subplot(gs[7, 1])
        ax.plot(indices_all, losses_all, color='lightcoral', alpha=0.5, linewidth=0.7)
        if len(losses_all) >= ma_window_all:
            ma_loss = moving_average_func(losses_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_loss, color='red', linewidth=2, label=f'MA({ma_window_all})')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss - Full Range ({len(losses_all)} batches)')
        min_loss_full = min(losses_all)
        max_loss_full = max(losses_all)
        padding = (max_loss_full - min_loss_full) * 0.05 if max_loss_full > min_loss_full else 0.1
        ax.set_ylim([max(0, min_loss_full - padding), max_loss_full + padding])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # CE Loss - Full Range
        ax = fig.add_subplot(gs[8, 0])
        ax.plot(indices_all, ce_losses_all, color='lightskyblue', alpha=0.5, linewidth=0.7)
        if len(ce_losses_all) >= ma_window_all:
            ma_ce = moving_average_func(ce_losses_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_ce, color='blue', linewidth=2, label=f'MA({ma_window_all})')
        ax.set_xlabel('Batch')
        ax.set_ylabel('CE Loss')
        ax.set_title(f'CE Loss - Full Range ({len(ce_losses_all)} batches)')
        min_ce_full = min(ce_losses_all)
        max_ce_full = max(ce_losses_all)
        padding = (max_ce_full - min_ce_full) * 0.05 if max_ce_full > min_ce_full else 0.1
        ax.set_ylim([max(0, min_ce_full - padding), max_ce_full + padding])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # KL Loss - Full Range
        ax = fig.add_subplot(gs[8, 1])
        ax.plot(indices_all, kl_losses_all, color='plum', alpha=0.5, linewidth=0.7)
        if len(kl_losses_all) >= ma_window_all:
            ma_kl = moving_average_func(kl_losses_all, ma_window_all)
            ax.plot(indices_all[ma_window_all-1:], ma_kl, color='purple', linewidth=2, label=f'MA({ma_window_all})')
        ax.set_xlabel('Batch')
        ax.set_ylabel('KL Loss')
        ax.set_title(f'KL Loss - Full Range ({len(kl_losses_all)} batches)')
        min_kl_full = min(kl_losses_all)
        max_kl_full = max(kl_losses_all)
        padding = (max_kl_full - min_kl_full) * 0.05 if max_kl_full > min_kl_full else 0.1
        ax.set_ylim([max(0, min_kl_full - padding), max_kl_full + padding])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ========== CONFUSION MATRICES (latest eval epoch) ==========

        confusion_categories = [
            (9, 0, 'tone'), (9, 1, 'sentiment'),
            (10, 0, 'safety'), (10, 1, 'toxicity'),
        ]

        latest_confusion = None
        if self.eval_epochs:
            latest_confusion = self.eval_epochs[-1].get('confusion_matrices')

        cmap = LinearSegmentedColormap.from_list('white_blue', ['white', '#4285f4'])

        for row, col, category in confusion_categories:
            ax = fig.add_subplot(gs[row, col])
            if latest_confusion and category in latest_confusion:
                matrix_data = latest_confusion[category]
                labels = list(matrix_data.keys())
                values = np.array([[matrix_data[true_val].get(pred_val, 0)
                                    for pred_val in labels] for true_val in labels])

                ax.imshow(values, cmap=cmap, aspect='auto')
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Ground Truth')

                for i in range(len(labels)):
                    for j in range(len(labels)):
                        count = values[i, j]
                        if count > 0:
                            color = 'white' if count > values.max() * 0.6 else 'black'
                            ax.text(j, i, str(int(count)), ha='center', va='center',
                                    fontsize=10, fontweight='bold', color=color)
            else:
                ax.text(0.5, 0.5, 'No confusion data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Confusion Matrix - {category.capitalize()} (Epoch {self.eval_epochs[-1]["epoch"] if self.eval_epochs else "?"})')

        save_path = Path(LOGS_DIR) / 'training_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if show:
            print(f"Saved: {save_path}")
