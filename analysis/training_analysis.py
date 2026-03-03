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

    def plot(self, moving_average_func, show=True):
        if not self.train_batches:
            print("No training data to plot.")
            return

        fig = plt.figure(figsize=(16, 62))
        gs = GridSpec(11, 2, figure=fig, hspace=0.35, wspace=0.25,
                      height_ratios=[0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        losses = [b['loss'] for b in self.train_batches]
        kl_losses = [b.get('kl_loss', 0) for b in self.train_batches]
        ce_losses = [b.get('ce_loss', 0) for b in self.train_batches]
        accuracies = [b['accuracy'] * 100 for b in self.train_batches]
        lrs = [b['learning_rate'] for b in self.train_batches]
        batch_indices = list(range(1, len(self.train_batches) + 1))

        # --- Summary panel ---
        self._plot_summary_panel(fig.add_subplot(gs[0, :]), accuracies, losses, ce_losses, kl_losses)

        # --- Metric time-series (rows 1-4: 90%-ylim left, recent-1000 right) ---
        metrics = [
            ('Accuracy', accuracies, 'lightgreen', 'green', '%'),
            ('Loss', losses, 'lightcoral', 'red', ''),
            ('CE Loss', ce_losses, 'lightskyblue', 'blue', ''),
            ('KL Loss', kl_losses, 'plum', 'purple', ''),
        ]

        ma_window = max(40, int(len(losses) * 0.03))
        last_90_start = int(len(losses) * 0.1)
        window_1000 = min(1000, len(losses))

        for row_offset, (name, data, light_color, dark_color, unit) in enumerate(metrics):
            row = row_offset + 1
            default_padding = 5 if name == 'Accuracy' else 0.1
            is_percent = name == 'Accuracy'
            data_90 = data[last_90_start:] if last_90_start < len(data) else data

            # Left: all data, y-axis from last 90%
            self._plot_timeseries(
                fig.add_subplot(gs[row, 0]), batch_indices, data, moving_average_func,
                light_color, dark_color, name, f'{name} - All ({len(data)} batches, y-axis: last 90%)',
                ma_window=ma_window, ylim_data=data_90, default_padding=default_padding,
                is_percent=is_percent, show_last_200_avg=True)

            # Right: recent window
            recent_data = data[-window_1000:]
            recent_indices = batch_indices[-window_1000:]
            self._plot_timeseries(
                fig.add_subplot(gs[row, 1]), recent_indices, recent_data, moving_average_func,
                light_color, dark_color, name, f'{name} - Recent ({window_1000} batches)',
                ma_window=40, ylim_data=recent_data, default_padding=default_padding,
                is_percent=is_percent, show_last_200_avg=True)

        # --- Epoch summaries (row 5) ---
        self._plot_epoch_loss(fig.add_subplot(gs[5, 0]))
        self._plot_epoch_accuracy(fig.add_subplot(gs[5, 1]))

        # --- LR and KL ratio schedules (row 6) ---
        ax = fig.add_subplot(gs[6, 0])
        ax.plot(batch_indices, lrs, color='orange', linewidth=2, label='Learning Rate')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

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

        # --- Full range charts (rows 7-8) ---
        for row_offset, (name, data, light_color, dark_color, unit) in enumerate(metrics):
            row = 7 + row_offset // 2
            col = row_offset % 2
            default_padding = 5 if name == 'Accuracy' else 0.1
            is_percent = name == 'Accuracy'

            self._plot_timeseries(
                fig.add_subplot(gs[row, col]), batch_indices, data, moving_average_func,
                light_color, dark_color, name, f'{name} - Full Range ({len(data)} batches)',
                ma_window=ma_window, ylim_data=data, default_padding=default_padding,
                is_percent=is_percent, show_last_200_avg=False, ylim_padding_ratio=0.05)

        # --- Confusion matrices (rows 9-10) ---
        self._plot_confusion_matrices(fig, gs)

        save_path = Path(LOGS_DIR) / 'training_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if show:
            print(f"Saved: {save_path}")

    def _plot_timeseries(self, ax, indices, data, moving_average_func,
                         light_color, dark_color, ylabel, title,
                         ma_window=40, ylim_data=None, default_padding=0.1,
                         is_percent=False, show_last_200_avg=True, ylim_padding_ratio=0.1):
        ax.plot(indices, data, color=light_color, alpha=0.5, linewidth=0.7)

        if len(data) >= ma_window:
            ma = moving_average_func(data, ma_window)
            ax.plot(indices[ma_window - 1:], ma, color=dark_color, linewidth=2, label=f'MA({ma_window})')

        if show_last_200_avg and len(data) >= 200:
            avg = np.mean(data[-200:])
            fmt = f'{avg:.1f}%' if is_percent else f'{avg:.4f}'
            ax.axhline(y=avg, color='black', linewidth=1, alpha=0.4, linestyle='-', label=f'Last 200 avg: {fmt}')

        if ylim_data is not None and len(ylim_data) > 0:
            min_val = min(ylim_data)
            max_val = max(ylim_data)
            padding = (max_val - min_val) * ylim_padding_ratio if max_val > min_val else default_padding
            low = max(0, min_val - padding)
            high = min(100, max_val + padding) if is_percent else max_val + padding
            ax.set_ylim([low, high])

        ax.set_xlabel('Batch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_summary_panel(self, ax, accuracies, losses, ce_losses, kl_losses):
        ax.axis('off')

        current_acc = np.mean(accuracies[-200:]) if len(accuracies) >= 200 else accuracies[-1] if accuracies else 0
        previous_acc = np.mean(accuracies[-400:-200]) if len(accuracies) >= 400 else (accuracies[0] if accuracies else 0)
        acc_diff = current_acc - previous_acc
        acc_trend = f"+{acc_diff:.1f}%" if acc_diff > 0 else f"{acc_diff:.1f}%" if acc_diff < 0 else "="

        current_loss = np.mean(losses[-200:]) if len(losses) >= 200 else losses[-1] if losses else 0
        previous_loss = np.mean(losses[-400:-200]) if len(losses) >= 400 else (losses[0] if losses else 0)
        loss_diff = current_loss - previous_loss
        loss_trend = f"{loss_diff:.4f}" if loss_diff < 0 else f"+{loss_diff:.4f}" if loss_diff > 0 else "="

        current_ce = np.mean(ce_losses[-200:]) if len(ce_losses) >= 200 else ce_losses[-1] if ce_losses else 0
        current_kl = np.mean(kl_losses[-200:]) if len(kl_losses) >= 200 else kl_losses[-1] if kl_losses else 0

        text = (
            f"Accuracy: {current_acc:.1f}% ({acc_trend})        "
            f"Loss: {current_loss:.4f} ({loss_trend})        "
            f"CE: {current_ce:.4f}        "
            f"KL: {current_kl:.4f}        "
            f"Batches: {len(self.train_batches):,}        "
            f"Epochs: {len(self.train_epochs)}"
        )
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    def _plot_epoch_loss(self, ax):
        if self.train_epochs:
            train_ep = [d['epoch'] for d in self.train_epochs]
            train_loss = [d['avg_loss'] for d in self.train_epochs]
            ax.plot(train_ep, train_loss, marker='o', color='red', linewidth=2, label='Train')
        if self.eval_epochs:
            eval_ep = [d['epoch'] for d in self.eval_epochs]
            eval_loss = [d['avg_loss'] for d in self.eval_epochs]
            ax.plot(eval_ep, eval_loss, marker='s', color='blue', linewidth=2, label='Eval')
        if not self.train_epochs and not self.eval_epochs:
            ax.text(0.5, 0.5, 'No epoch data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Epoch Summary - Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_epoch_accuracy(self, ax):
        if self.eval_epochs:
            eval_ep = [d['epoch'] for d in self.eval_epochs]
            series = [
                ('Teacher-Forced', 'green', 'o', lambda d: d.get('teacher_forced_accuracy', d.get('accuracy', 0))),
                ('Student', 'blue', 's', lambda d: d.get('student_accuracy', 0)),
                ('Classification', 'orange', '^', lambda d: d.get('classification_accuracy', 0)),
            ]
            all_values = []
            for label, color, marker, getter in series:
                values = [getter(d) * 100 for d in self.eval_epochs]
                ax.plot(eval_ep, values, marker=marker, color=color, linewidth=2, label=label)
                all_values.extend(values)
            if all_values:
                padding = (max(all_values) - min(all_values)) * 0.1 if max(all_values) > min(all_values) else 5
                ax.set_ylim([max(0, min(all_values) - padding), min(100, max(all_values) + padding)])
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No evaluation data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Epoch Summary - Eval Accuracy')
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrices(self, fig, gs):
        layout = [(9, 0, 'tone'), (9, 1, 'sentiment'), (10, 0, 'safety'), (10, 1, 'toxicity')]

        latest_confusion = None
        if self.eval_epochs:
            latest_confusion = self.eval_epochs[-1].get('confusion_matrices')

        cmap = LinearSegmentedColormap.from_list('white_blue', ['white', '#4285f4'])

        for row, col, category in layout:
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

            epoch_label = self.eval_epochs[-1]['epoch'] if self.eval_epochs else '?'
            ax.set_title(f'Confusion Matrix - {category.capitalize()} (Epoch {epoch_label})')
