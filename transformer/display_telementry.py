import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class TelemetryVisualizer:
    def __init__(self, telemetry_file="telemetry/training.jsonl"):
        self.telemetry_file = Path(telemetry_file)
        self.data = self._load_data()

    def _load_data(self):
        if not self.telemetry_file.exists():
            print(f"No telemetry file found at {self.telemetry_file}")
            return []

        data = []
        with open(self.telemetry_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        print(f"Loaded {len(data)} telemetry entries")
        return data

    def plot_training_progress(self):
        train_examples = [d for d in self.data if d['type'] == 'train_example']
        train_epochs = [d for d in self.data if d['type'] == 'train_epoch']
        eval_epochs = [d for d in self.data if d['type'] == 'eval_epoch']

        if not train_examples:
            print("No training data found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Telemetry', fontsize=16, fontweight='bold')

        self._plot_example_losses(axes[0, 0], train_examples)
        self._plot_epoch_summary(axes[0, 1], train_epochs, eval_epochs)
        self._plot_loss_distribution(axes[1, 0], train_examples)
        self._plot_accuracy_progress(axes[1, 1], eval_epochs)

        plt.tight_layout()
        plt.savefig('telemetry/training_progress.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved visualization to telemetry/training_progress.png")
        plt.show()

    def _plot_example_losses(self, ax, train_examples):
        epochs = defaultdict(list)

        for entry in train_examples:
            epoch = entry['epoch']
            loss = entry['loss']
            epochs[epoch].append(loss)

        for epoch, losses in sorted(epochs.items()):
            ax.plot(losses, alpha=0.7, label=f'Epoch {epoch}')

        ax.set_xlabel('Example Index (within batch)')
        ax.set_ylabel('Loss')
        ax.set_title('Per-Example Loss During Training')
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

    def _plot_loss_distribution(self, ax, train_examples):
        epochs = defaultdict(list)

        for entry in train_examples:
            epoch = entry['epoch']
            loss = entry['loss']
            epochs[epoch].append(loss)

        epoch_nums = sorted(epochs.keys())
        loss_data = [epochs[e] for e in epoch_nums]

        bp = ax.boxplot(loss_data, labels=[
                        f'E{e}' for e in epoch_nums], patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Distribution per Epoch')
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


visualizer = TelemetryVisualizer()
visualizer.print_summary()
visualizer.plot_training_progress()
