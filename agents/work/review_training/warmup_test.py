import os, sys, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
exec(open("resume_test.py").read().split("shutil.rmtree")[0])   # reuse harness definitions only
import torch.optim
recorded = []
original_step = torch.optim.AdamW.step
def recording_step(self, *args, **kwargs):
    recorded.append(self.param_groups[0]["lr"])
    return original_step(self, *args, **kwargs)
torch.optim.AdamW.step = recording_step
shutil.rmtree(os.path.join(RUNS_ROOT, "warm"), ignore_errors=True)
import dataclasses
_old_spec = spec
spec = lambda name: dataclasses.replace(_old_spec(name), lr_warmup_ratio=0.25)
run("warm")
per_batch = [recorded[i * 32] for i in range(len(recorded) // 32)]
print("optimizer steps:", len(recorded), " LR actually used at the first step of each training batch:", per_batch)
