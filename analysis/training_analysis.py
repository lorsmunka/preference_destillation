import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
            print(f"  Accuracy: {latest_eval['accuracy'] * 100:.1f}%")

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

    def plot(self, moving_average_func, exponential_moving_average_func):
        if not self.train_batches:
            print("No training data to plot.")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')

        losses = [b['loss'] for b in self.train_batches]
        accuracies = [b['accuracy'] * 100 for b in self.train_batches]
        lrs = [b['learning_rate'] for b in self.train_batches]
        batch_indices = list(range(1, len(self.train_batches) + 1))

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(batch_indices, losses, color='lightcoral', alpha=0.4, linewidth=0.5)
        ema = exponential_moving_average_func(losses, alpha=0.05)
        ax.plot(batch_indices, ema, color='red', linewidth=2, label='EMA (α=0.05)')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss - Full History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        window_size = min(200, len(losses))
        if len(losses) > window_size:
            recent_losses = losses[-window_size:]
            recent_indices = batch_indices[-window_size:]
            ax.plot(recent_indices, recent_losses, color='lightcoral', alpha=0.6, linewidth=1)
            if len(recent_losses) >= 20:
                smoothed = moving_average_func(recent_losses, 20)
                ax.plot(recent_indices[:len(smoothed)], smoothed, color='red', linewidth=2, label='MA(20)')
                ax.legend()
        else:
            ax.plot(batch_indices, losses, color='red', linewidth=1)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss - Recent ({window_size} batches)')
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(batch_indices, accuracies, color='lightgreen', alpha=0.4, linewidth=0.5)
        ema_acc = exponential_moving_average_func(accuracies, alpha=0.05)
        ax.plot(batch_indices, ema_acc, color='green', linewidth=2, label='EMA (α=0.05)')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy - Full History')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(batch_indices, lrs, color='orange', linewidth=2, label='Learning Rate')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate', color='orange')
        ax.tick_params(axis='y', labelcolor='orange')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)

        kl_ratios = [b.get('kl_ratio', 0) for b in self.train_batches]
        if any(kl_ratios):
            ax2 = ax.twinx()
            ax2.plot(batch_indices, kl_ratios, color='purple', linewidth=2, label='KL Ratio')
            ax2.set_ylabel('KL Ratio', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.set_ylim([0, 1])
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        ax.set_title('LR & KL Ratio Schedule')

        ax = fig.add_subplot(gs[2, 0])
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

        ax = fig.add_subplot(gs[2, 1])
        if self.eval_epochs:
            eval_ep = [d['epoch'] for d in self.eval_epochs]
            eval_acc = [d['accuracy'] * 100 for d in self.eval_epochs]
            ax.plot(eval_ep, eval_acc, marker='o', color='green', linewidth=2)
            ax.set_ylim([0, 100])
        else:
            ax.text(0.5, 0.5, 'No evaluation data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Epoch Summary - Eval Accuracy')
        ax.grid(True, alpha=0.3)

        save_path = Path(LOGS_DIR) / 'training_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
