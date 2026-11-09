"""Brings the rented machines' speed results home: speed_benchmark.py ran on each, their CSVs were copied to
saves/speed_pods/<machine>/, and this merges them into experiments/results/speed.csv and speed_replies.csv (the
keys start with the machine, so nothing is overwritten). The machines keep the names torch / the OS reported
("x86_64" on the Linux CPU pod); the thesis figure and table give them readable names.

    python experiments/speed_merge.py"""

import csv

from _common import REPO_ROOT, write_rows
from speed_benchmark import REPLY_COLUMNS, SUMMARY_COLUMNS

STAGED = REPO_ROOT / "saves" / "speed_pods"


def main():
    for folder in sorted(path for path in STAGED.iterdir() if path.is_dir()):
        for name, columns in (("speed.csv", SUMMARY_COLUMNS), ("speed_replies.csv", REPLY_COLUMNS)):
            if (folder / name).exists():
                with open(folder / name, encoding="utf-8", newline="") as file:
                    rows = list(csv.DictReader(file))
                write_rows(name, columns, rows)
                print(f"{folder.name}/{name}: {len(rows)} rows merged")


if __name__ == "__main__":
    main()
