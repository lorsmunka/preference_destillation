"""Which batch files a run trains on and which it is tested on, and reading them.

`split_batch_files` is the one place the train/test rule lives; analysis code calls the
same function, so a re-evaluation always scores a run on its own test set.
"""

import random
from pathlib import Path
from time import time
from typing import List, Optional, Sequence, Tuple

from library.shared.batch_files import (batch_paths, read_batch, read_batch_meta,
                                        vocabulary_fingerprint)
from library.specs import DEFAULT_MAX_TEST_EXAMPLES

TRAIN = "train"
TEST = "test"


def split_batch_files(batches_dir, training_test_ratio: float,
                      max_training_examples: Optional[int] = None, batch_size: int = 32,
                      max_test_examples: Optional[int] = DEFAULT_MAX_TEST_EXAMPLES,
                      training_groups: Optional[Sequence[str]] = None) -> Tuple[List[Path], List[Path]]:
    """The run's (training_files, test_files): two lists of batch-file Paths in batch order.

        training_files, test_files = split_batch_files(
            batches_dir, training_test_ratio, max_training_examples=None,
            batch_size=32, max_test_examples=1024)

    The test set is a fixed tail of the whole directory, so it stays the same whatever
    max_training_examples is: the last (1 - training_test_ratio) of the files form the test
    region, and the test set is the last max_test_examples // batch_size files of that
    region (None = the whole region). Training uses the files before the region, capped at
    the first max_training_examples // batch_size of them (None = all). Only training is
    capped by max_training_examples, and the two never share a file.

    When the batches carry per-record "split" labels ("train" / "test"), the labels decide
    instead of the position: a file is a training file if it holds a "train" record and a
    test file if it holds a "test" record (the same caps apply). A file holding both is in
    both lists, so read every file through select_records(records, role), which keeps only
    the records of that role (BatchHandler does this).

    training_groups (e.g. ("original", "new")) keeps only the training files holding a record
    with one of those "group" flags, before the cap; BatchHandler then trains on those records
    only. The test set never depends on it.
    """
    all_files = batch_paths(batches_dir)
    test_region_start = int(len(all_files) * training_test_ratio)

    if _carries_split_labels(all_files):
        file_roles = [_labelled_file_roles(path, index >= test_region_start)
                      for index, path in enumerate(all_files)]
    else:
        file_roles = [{TEST} if index >= test_region_start else {TRAIN} for index in range(len(all_files))]

    training_files = [path for path, roles in zip(all_files, file_roles) if TRAIN in roles]
    test_files = [path for path, roles in zip(all_files, file_roles) if TEST in roles]

    if training_groups is not None:
        training_files = [path for path in training_files if set(_record_groups(path)) & set(training_groups)]
    if max_training_examples is not None:
        training_files = training_files[:max_training_examples // batch_size]
    if max_test_examples is not None:
        test_file_count = max_test_examples // batch_size
        test_files = test_files[max(len(test_files) - test_file_count, 0):]
    return training_files, test_files


def select_records(records: List[dict], role: str) -> List[dict]:
    """The records of one batch file that belong to `role` ("train" or "test").

    A file without "split" labels belongs wholly to the role split_batch_files gave it, so
    all its records are kept. In a labelled file only the records labelled `role` are kept
    (an unlabelled record next to labelled ones is ambiguous and used for neither)."""
    if not any(record.get("split") is not None for record in records):
        return records
    return [record for record in records if record.get("split") == role]


def _record_splits(path) -> List[Optional[str]]:
    """The "split" label of every record in one batch file (None where it has none).
    For an .npz batch this reads only the small JSON blob, not the logits."""
    meta = read_batch_meta(path)
    if meta:
        return meta.get("split") or [None] * len(meta.get("sentences", []))
    return [record.get("split") for record in read_batch(path)]   # legacy .jsonl


def _record_groups(path) -> List[Optional[str]]:
    """The "group" flag of every record in one batch file (None where it has none)."""
    meta = read_batch_meta(path)
    if meta:
        return meta.get("group") or [None] * len(meta.get("sentences", []))
    return [record.get("group") for record in read_batch(path)]   # legacy .jsonl


def _carries_split_labels(all_files: List[Path]) -> bool:
    """Whether the batches carry per-record split labels. Corpora that have them write
    their train records first and test records last, so the first and last files tell."""
    probe_files = all_files[:1] + all_files[-1:]
    return any(split is not None for path in probe_files for split in _record_splits(path))


def _labelled_file_roles(path, in_test_region: bool) -> set:
    splits = set(_record_splits(path))
    if splits <= {None}:   # this file has no labels: its position decides, as without labels
        return {TEST if in_test_region else TRAIN}
    return splits & {TRAIN, TEST}


class BatchHandler:
    def __init__(self, batches_dir: str, training_test_ratio: float, max_training_examples: int = None,
                 batch_size: int = 32, max_test_examples: Optional[int] = DEFAULT_MAX_TEST_EXAMPLES,
                 teacher_token_list: Optional[List[str]] = None, shuffle_seed: Optional[int] = None,
                 training_groups: Optional[Sequence[str]] = None):
        self.batches_dir = batches_dir
        self.training_groups = training_groups
        self.training_files, self.test_files = split_batch_files(
            batches_dir, training_test_ratio, max_training_examples, batch_size, max_test_examples, training_groups)
        # The domain's full reduced vocabulary (auxiliary percentage 1.0): the tokens the
        # batches' logit columns must be, checked against each batch's vocabulary_sha1.
        self.expected_vocabulary_sha1 = (vocabulary_fingerprint(teacher_token_list)
                                         if teacher_token_list is not None else None)
        self.checked_files = set()
        self.shuffle_seed = shuffle_seed

    def training_batch_count(self) -> int:
        return len(self.training_files)

    def test_batch_count(self) -> int:
        return len(self.test_files)

    def training_order(self, epoch: int) -> List[int]:
        """The order the training files are visited in this epoch: file order, or a
        permutation fixed by (shuffle_seed, epoch), so a resumed epoch replays the same one."""
        order = list(range(len(self.training_files)))
        if self.shuffle_seed is not None:
            random.Random(f"{self.shuffle_seed}-{epoch}").shuffle(order)
        return order

    def get_training_batch(self, epoch: int, position: int) -> List[dict]:
        """The training examples of the epoch's `position`-th batch (0-based)."""
        records = self._read(self.training_files[self.training_order(epoch)[position]], TRAIN)
        if self.shuffle_seed is not None:
            random.Random(f"{self.shuffle_seed}-{epoch}-{position}").shuffle(records)
        return records

    def get_test_batch(self, index: int) -> List[dict]:
        return self._read(self.test_files[index], TEST)

    def _read(self, path: Path, role: str) -> List[dict]:
        print(f"Loading {path.name}...")
        start_time = time()
        self._check_vocabulary(path)
        records = select_records(read_batch(path), role)
        if role == TRAIN and self.training_groups is not None:
            records = [record for record in records if record.get("group") in self.training_groups]
        print(f"Loaded {path.name} -> took {time() - start_time:.2f} seconds.")
        return records

    def _check_vocabulary(self, path: Path) -> None:
        if self.expected_vocabulary_sha1 is None or path in self.checked_files:
            return
        stored_sha1 = read_batch_meta(path).get("vocabulary_sha1")   # None in batches made before it existed: unknown, not an error
        if stored_sha1 is not None and stored_sha1 != self.expected_vocabulary_sha1:
            raise ValueError(
                f"{path}: its logit columns belong to a different reduced vocabulary "
                f"(vocabulary_sha1 {stored_sha1}) than the model's ({self.expected_vocabulary_sha1}). "
                f"The output vocabulary was rebuilt or reordered after this batch was generated, so "
                f"its KL targets would land on the wrong tokens. Restore the vocabulary the batches "
                f"were made with, or regenerate the batches.")
        self.checked_files.add(path)
