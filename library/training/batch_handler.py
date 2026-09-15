from time import time

from library.shared.batch_files import batch_count, batch_path, read_batch


class BatchHandler:
    def __init__(self, batches_dir: str, training_test_ratio: float, max_training_examples: int = None, batch_size: int = 32):
        self.batches_dir = batches_dir
        self.training_test_ratio = training_test_ratio
        self.max_training_examples = max_training_examples
        self.batch_size = batch_size

    def get_batch(self, index):
        batch_index = index + 1
        print(f"Loading batch {batch_index}...")
        start_time = time()

        batch = read_batch(batch_path(self.batches_dir, batch_index))

        elapsed_time = time() - start_time
        print(
            f"Loaded batch {batch_index} -> took {elapsed_time:.2f} seconds.")

        return batch

    def get_batch_count(self):
        return batch_count(self.batches_dir)

    def _effective_batch_count(self):
        total_batches = self.get_batch_count()
        if self.max_training_examples is not None:
            max_batches = self.max_training_examples // self.batch_size
            return min(total_batches, max_batches)
        return total_batches

    def get_training_batches_radius(self):
        effective = self._effective_batch_count()
        return 0, int(effective * self.training_test_ratio)

    def get_test_batches_radius(self):
        effective = self._effective_batch_count()
        return int(effective * self.training_test_ratio), effective
