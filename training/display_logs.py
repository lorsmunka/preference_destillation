import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from shared import LOGS_DIR


class LogVisualizer:
    def __init__(self, log_file=f"{LOGS_DIR}/training.jsonl", smooth_window=200):
        self.log_file = Path(log_file)
        self.smooth_window = smooth_window
        self.data = self._load_data()

    def _load_data(self):
        if not self.log_file.exists():
            print(f"No log file found at {self.log_file}")
            return []
        with open(self.log_file, 'r') as file:
            return [json.loads(line.strip()) for line in file]

    def plot_training_progress(self):
        train_examples = [d for d in self.data if d['type'] == 'train_example']
        train_epochs = [d for d in self.data if d['type'] == 'train_epoch']
        eval_epochs = [d for d in self.data if d['type'] == 'eval_epoch']

        if not train_examples:
            print("No training data found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Logs (Continuous View)',
                     fontsize=16, fontweight='bold')

        self._plot_continuous_loss(axes[0, 0], train_examples)
        self._plot_epoch_summary(axes[0, 1], train_epochs, eval_epochs)
        self._plot_loss_histogram(axes[1, 0], train_examples)
        self._plot_accuracy_progress(axes[1, 1], eval_epochs)

        plt.tight_layout()
        plt.savefig(f'{LOGS_DIR}/training_progress_continuous.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {LOGS_DIR}/training_progress_continuous.png")
        plt.show()

    def _smooth(self, values):
        if len(values) < self.smooth_window:
            return values
        kernel = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(values, kernel, mode='valid')

    def _plot_continuous_loss(self, ax, train_examples):
        losses = [d['loss'] for d in train_examples]
        smoothed = self._smooth(losses)

        ax.plot(losses, color='lightgray', alpha=0.4, label='Raw Loss')
        ax.plot(np.arange(len(smoothed)), smoothed, color='red',
                linewidth=2, label=f'Smoothed (window={self.smooth_window})')

        ax.set_xlabel('Example Index (global)')
        ax.set_ylabel('Loss')
        ax.set_title('Continuous Training Loss (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_epoch_summary(self, ax, train_epochs, eval_epochs):
        if train_epochs:
            epochs = [d['epoch'] for d in train_epochs]
            train_losses = [d['avg_loss'] for d in train_epochs]
            ax.plot(epochs, train_losses, marker='o',
                    label='Train Loss', linewidth=2)

        if eval_epochs:
            epochs = [d['epoch'] for d in eval_epochs]
            eval_losses = [d['avg_loss'] for d in eval_epochs]
            ax.plot(epochs, eval_losses, marker='s',
                    label='Eval Loss', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.set_title('Epoch-Level Loss Summary')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_loss_histogram(self, ax, train_examples):
        losses = [d['loss'] for d in train_examples]
        ax.hist(losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Loss')
        ax.set_ylabel('Frequency')
        ax.set_title('Loss Distribution (All Examples)')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_accuracy_progress(self, ax, eval_epochs):
        if not eval_epochs:
            ax.text(0.5, 0.5, 'No evaluation data yet',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Evaluation Accuracy')
            return

        epochs = [d['epoch'] for d in eval_epochs]
        accuracies = [d['accuracy'] * 100 for d in eval_epochs]
        ax.plot(epochs, accuracies, marker='o', linewidth=2, color='green')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Evaluation Accuracy Progress')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

    def print_summary(self):
        train_examples = [d for d in self.data if d['type'] == 'train_example']
        train_epochs = [d for d in self.data if d['type'] == 'train_epoch']
        eval_epochs = [d for d in self.data if d['type'] == 'eval_epoch']

        print("\n=== Training Summary ===")
        print(f"Total examples trained: {len(train_examples)}")
        print(f"Epochs completed: {len(train_epochs)}")
        print(f"Evaluations completed: {len(eval_epochs)}")

        if train_examples:
            first_loss = train_examples[0]['loss']
            last_loss = train_examples[-1]['loss']
            print(f"\nFirst example loss: {first_loss:.4f}")
            print(f"Last example loss: {last_loss:.4f}")
            print(
                f"Improvement: {((first_loss - last_loss) / first_loss * 100):.1f}%")

        if eval_epochs:
            print(
                f"\nLatest eval accuracy: {eval_epochs[-1]['accuracy']*100:.2f}%")
            print(f"Latest eval loss: {eval_epochs[-1]['avg_loss']:.4f}")


visualizer = LogVisualizer()
visualizer.print_summary()
visualizer.plot_training_progress()
