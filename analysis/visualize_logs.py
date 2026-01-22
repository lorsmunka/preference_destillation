from analysis.training_analysis import TrainingAnalyzer
from analysis.generation_analysis import GenerationAnalyzer
from shared import LOGS_DIR
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


GENERATION_LOG_FILE = Path(LOGS_DIR) / "generation.jsonl"
TRAINING_LOG_FILE = Path(LOGS_DIR) / "training.jsonl"


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]


def moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


def exponential_moving_average(values: List[float], alpha: float = 0.1) -> np.ndarray:
    ema = np.zeros(len(values))
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


class LogVisualizer:
    def __init__(self):
        generation_data = load_jsonl(GENERATION_LOG_FILE)
        training_data = load_jsonl(TRAINING_LOG_FILE)
        self.generation_analyzer = GenerationAnalyzer(generation_data)
        self.training_analyzer = TrainingAnalyzer(training_data)

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
                    moving_average, exponential_moving_average)
            elif choice == '5':
                self.generation_analyzer.print_summary()
                self.training_analyzer.print_summary()
                self.generation_analyzer.plot(moving_average)
                self.training_analyzer.plot(
                    moving_average, exponential_moving_average)
            elif choice == 'q':
                print("Bye!")
                break
            else:
                print("Invalid option")


def main():
    visualizer = LogVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
