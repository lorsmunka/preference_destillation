import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import LOGS_DIR


class GenerationAnalyzer:
    def __init__(self, data: List[Dict]):
        self.batches = [d for d in data if d.get('type') == 'batch']

    def print_summary(self):
        print("\n" + "=" * 50)
        print("GENERATION SUMMARY")
        print("=" * 50)

        if not self.batches:
            print("No generation data found.")
            return

        total_processed = sum(b['processed'] for b in self.batches)
        total_successful = sum(b['successful'] for b in self.batches)
        total_skipped = sum(b['skipped'] for b in self.batches)
        total_time = sum(b['time_seconds'] for b in self.batches)

        skip_reasons_total = {}
        for batch in self.batches:
            for reason, count in batch.get('skip_reasons', {}).items():
                skip_reasons_total[reason] = skip_reasons_total.get(reason, 0) + count

        sessions = set(b.get('session_id', 'unknown') for b in self.batches)

        print(f"\nBatches completed: {len(self.batches)}")
        print(f"Sessions: {len(sessions)}")
        print(f"\nSentences processed: {total_processed:,}")
        print(f"Sentences successful: {total_successful:,}")
        print(f"Sentences skipped: {total_skipped:,}")
        print(f"Success rate: {(total_successful / total_processed * 100) if total_processed > 0 else 0:.1f}%")

        print(f"\nTotal time: {total_time / 3600:.2f} hours")
        print(f"Avg time per batch: {total_time / len(self.batches):.1f}s")
        print(f"Avg time per sentence: {total_time / total_processed:.2f}s")

        if skip_reasons_total:
            print("\nSkip reasons:")
            for reason, count in sorted(skip_reasons_total.items(), key=lambda x: -x[1]):
                pct = count / total_skipped * 100 if total_skipped > 0 else 0
                print(f"  {reason}: {count:,} ({pct:.1f}%)")

        if len(self.batches) >= 2:
            recent_batches = self.batches[-10:]
            recent_time = sum(b['time_seconds'] for b in recent_batches)
            recent_processed = sum(b['processed'] for b in recent_batches)
            print(f"\nRecent pace (last {len(recent_batches)} batches): {recent_time / recent_processed:.2f}s per sentence")

    def plot(self, moving_average_func):
        if not self.batches:
            print("No generation data to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data Generation Progress', fontsize=14, fontweight='bold')

        batch_indices = list(range(1, len(self.batches) + 1))
        times = [b['time_seconds'] for b in self.batches]
        success_rates = [b['successful'] / b['processed'] * 100 for b in self.batches]
        cumulative_successful = np.cumsum([b['successful'] for b in self.batches])

        ax = axes[0, 0]
        ax.plot(batch_indices, times, color='steelblue', alpha=0.6, linewidth=1)
        if len(times) >= 10:
            smoothed = moving_average_func(times, min(20, len(times) // 5))
            ax.plot(range(len(smoothed)), smoothed, color='darkblue', linewidth=2, label='Moving avg')
            ax.legend()
        ax.set_xlabel('Batch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time per Batch')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(batch_indices, success_rates, color='green', alpha=0.6, linewidth=1)
        if len(success_rates) >= 10:
            smoothed = moving_average_func(success_rates, min(20, len(success_rates) // 5))
            ax.plot(range(len(smoothed)), smoothed, color='darkgreen', linewidth=2, label='Moving avg')
            ax.legend()
        ax.set_xlabel('Batch')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate per Batch')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(batch_indices, cumulative_successful, color='purple', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Cumulative Successful')
        ax.set_title('Cumulative Successful Sentences')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        skip_reasons_total = {}
        for batch in self.batches:
            for reason, count in batch.get('skip_reasons', {}).items():
                skip_reasons_total[reason] = skip_reasons_total.get(reason, 0) + count
        if skip_reasons_total:
            reasons = list(skip_reasons_total.keys())
            counts = list(skip_reasons_total.values())
            ax.barh(reasons, counts, color='coral')
            ax.set_xlabel('Count')
            ax.set_title('Skip Reasons')
        else:
            ax.text(0.5, 0.5, 'No skipped sentences', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Skip Reasons')

        plt.tight_layout()
        save_path = Path(LOGS_DIR) / 'generation_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
