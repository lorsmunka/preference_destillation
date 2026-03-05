import json
from pathlib import Path
from typing import List, Dict

import numpy as np

from analysis.generation_analysis import GenerationAnalyzer
from analysis.training_analysis import TrainingAnalyzer
from shared import LOGS_DIR


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]


def moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


class LogVisualizer:
    def __init__(self, logs_dir: str = LOGS_DIR):
        generation_log_file = Path(logs_dir) / "generation.jsonl"
        training_log_file = Path(logs_dir) / "training.jsonl"
        generation_data = load_jsonl(generation_log_file)
        training_data = load_jsonl(training_log_file)
        self.generation_analyzer = GenerationAnalyzer(generation_data)
        self.training_analyzer = TrainingAnalyzer(training_data, logs_dir)

    def run(self):
        print("\n" + "=" * 50)
        print("LOG VISUALIZER")
        print("=" * 50)

        while True:
            print("\nOptions:")
            print("  [1] Generation statistics")
            print("  [2] Training statistics")
            print("  [3] Generation charts")
            print("  [4] Training charts")
            print("  [5] Full report (all stats + charts)")
            print("  [q] Quit")

            choice = input("\nSelect option: ").strip().lower()

            if choice == '1':
                self.generation_analyzer.print_summary()
            elif choice == '2':
                self.training_analyzer.print_summary()
            elif choice == '3':
                self.generation_analyzer.plot(moving_average)
            elif choice == '4':
                self.training_analyzer.plot(
                    moving_average)
            elif choice == '5':
                self.generation_analyzer.print_summary()
                self.training_analyzer.print_summary()
                self.generation_analyzer.plot(moving_average)
                self.training_analyzer.plot(
                    moving_average)
            elif choice == 'q':
                print("Bye!")
                break
            else:
                print("Invalid option")


def update_training_plot(logs_dir: str = LOGS_DIR):
    from time import time
    start_time = time()

    training_log_file = Path(logs_dir) / "training.jsonl"
    print("Updating training_progress.png...")
    training_data = load_jsonl(training_log_file)
    if not training_data:
        return

    analyzer = TrainingAnalyzer(training_data, logs_dir)
    analyzer.plot(moving_average, show=False)
    elapsed = time() - start_time
    print(f"training_progress.png updated -> took {elapsed:.2f}s\n")


def main():
    visualizer = LogVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
