"""Simulated crash during generation: does a resume pick up where batches stopped?"""
import sys, os, json, shutil
sys.path.insert(0, r"D:\preference_destillation")
from library.data_gen.generator import DistillationDataGenerator
from library.data_gen.saving_handler import SavingHandler
from library.shared.logging import RunLogger
from library.shared.batch_files import batch_paths
from library.domain import MATH_TUTOR

OUT = r"C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_datagen\resume_out"
shutil.rmtree(OUT, ignore_errors=True)

class FakeInputs:
    input_count = 200
    def get_input(self, i): return f"text {i}"
    def get_teacher_mode(self, i): return None
    def get_target(self, i): return None
    def get_ground_truth(self, i): return None

class FakeModel:
    def __init__(self, crash_at=None): self.crash_at, self.calls = crash_at, 0
    def generate_training_examples(self, texts):
        self.calls += 1
        if self.crash_at and self.calls == self.crash_at: raise RuntimeError("CUDA out of memory (simulated)")
        return [({"sentence": t, "model_response": "1;", "steps": [{"token": "1", "logits": [0.0, 1.0], "predicted_token_index": 1}]}, None) for t in texts]

class NoExit:
    def check_exit(self): return False

def make(model):
    g = DistillationDataGenerator.__new__(DistillationDataGenerator)
    g.config = {"domain": "math_tutor", "model_name": "fake", "description": "", "max_examples": 1000, "batch_size": 32}
    g.domain = MATH_TUTOR; g.exit_listener = NoExit(); g.model_name = "fake"
    g.max_examples, g.batch_size, g.generation_batch_size = 1000, 32, 16
    g.input_handler = FakeInputs(); g.model_handler = model
    g.logger = RunLogger(OUT, batch_size=32); g.saving_handler = SavingHandler(g.logger, OUT)
    return g

try:
    make(FakeModel(crash_at=10)).run()
except RuntimeError as error:
    print("session 1 crashed:", error)
print("batch files on disk:", [p.name for p in batch_paths(OUT)])
print("state.json exists:", os.path.exists(os.path.join(OUT, "state.json")))
resumed = make(FakeModel())
print("resume starts at processed =", resumed.logger.processed_sentence_count, "successful =", resumed.logger.successful_sentence_count)
