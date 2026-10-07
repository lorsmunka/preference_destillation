import sys, torch
sys.path.insert(0, r"D:\preference_destillation")
from library import RunStore
from library.model.student import StudentModel
store = RunStore()
run = store.get("ideaA-vocab-aux000-s1")
label, path = store.resolve_checkpoint(run.name, "latest")
student = StudentModel.from_run(run, path, device="cpu")
state = dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
for k in ("rotary_embedding.cos_cached", "rotary_embedding.sin_cached"): state.pop(k, None)
result = student.model.load_state_dict(state, strict=False)
print("checkpoint", label, "missing:", result.missing_keys, "unexpected:", result.unexpected_keys)
