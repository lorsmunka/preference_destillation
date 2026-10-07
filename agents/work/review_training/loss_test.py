import math
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, r"D:\preference_destillation")
import torch
import torch.nn.functional as F
from library.training.losses import DistillationLoss
from library.training.schedules import Schedules

torch.manual_seed(0)
steps, vocab = 5, 7
student = torch.randn(steps, vocab, dtype=torch.float64)
teacher = torch.randn(steps, vocab, dtype=torch.float64) * 3
target = teacher.argmax(-1)

def manual(divergence, s, t, ts, tt):
    p = F.softmax(t / tt, -1)
    q = F.softmax(s / ts, -1)
    if divergence == "forward_kl":
        return (p * (p.log() - q.log())).sum(-1)
    if divergence == "reverse_kl":
        return (q * (q.log() - p.log())).sum(-1)
    if divergence == "js":
        m = 0.5 * (p + q)
        return 0.5 * (p * (p.log() - m.log())).sum(-1) + 0.5 * (q * (q.log() - m.log())).sum(-1)
    if divergence == "tvd":
        return 0.5 * (p - q).abs().sum(-1)

for divergence in ["forward_kl", "reverse_kl", "js", "tvd"]:
    for ts, tt in [(1.0, 1.0), (2.0, 2.0), (1.0, 3.0)]:
        loss = DistillationLoss(divergence)
        d, ce, total = loss.compute(student, teacher, target, ts, tt, kl_ratio=0.3, reduction="sum")
        expected_d = (manual(divergence, student, teacher, ts, tt) * ts ** 2).sum()
        expected_ce = (F.cross_entropy(student / ts, target, reduction="none") * ts ** 2).sum()
        expected_total = 0.3 * expected_d + 0.7 * expected_ce
        print(f"{divergence:10} Ts={ts} Tt={tt}: |d err|={abs(d - expected_d).item():.2e} "
              f"|ce err|={abs(ce - expected_ce).item():.2e} |total err|={abs(total - expected_total).item():.2e}")

# reverse KL with a near one-hot teacher whose tail is below the 1e-12 clamp (float32, as in training)
teacher_sharp = torch.tensor([[40.0, 0.0, -10.0, -30.0]])
student_flat = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
loss = DistillationLoss("reverse_kl")
d, _, _ = loss.compute(student_flat, teacher_sharp, torch.tensor([0]), 1.0, 1.0, 1.0, "sum")
true_value = manual("reverse_kl", student_flat.double(), teacher_sharp.double(), 1.0, 1.0).sum()
print(f"reverse_kl sharp teacher: code={d.item():.3f} true={true_value.item():.3f}")

# entropy weighting: normalized entropy in [0, 1]
loss = DistillationLoss("forward_kl", entropy_kl_weighting=1.0)
uniform = torch.zeros(1, 525)
print("entropy weight on uniform teacher:", loss.kl_ratio_per_token(F.softmax(uniform, -1), 0.5).item())

# schedules
config = dict(learning_rate=1e-3, lr_warmup_ratio=0.1, kl_ratio_start=0.9, kl_ratio_end=0.1,
              distillation_temperature_start=1.0, distillation_temperature_end=1.0)
schedule = Schedules(config, steps_per_epoch=50, total_training_steps=150)
print("lr at steps 0,1,15,16,149,150:", [round(schedule.learning_rate(s), 6) for s in (0, 1, 15, 16, 149, 150)])
print("kl at steps 0,75,149,150:", [round(schedule.kl_ratio(s), 4) for s in (0, 75, 149, 150)])
