import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, r"D:/preference_destillation")
from pathlib import Path
import torch
import torch.nn.functional as F
from library.tooling.run import Run
from library.model.student import StudentModel
from library.shared.batch_files import read_batch

RUNS = Path(r"D:/preference_destillation/runs")
def load_run(name):
    info = json.load(open(RUNS / name / "info.json"))
    return Run(name, info, RUNS / name)

records = read_batch(r"D:/preference_destillation/batches/post_generation/google_gemma-3-4b-it/batch_97.npz")
example = records[0]
print("SENTENCE:", example["sentence"][:300])
print("RESPONSE:", example["model_response"][:300])
N = 14
out = {"sentence": example["sentence"], "response": example["model_response"], "rows": {}}
for run_name, epochs in [("e01-postgen-pure-kl-s1", [0, 9]), ("e01-postgen-pure-ce-s1", [9])]:
    run = load_run(run_name)
    for epoch in epochs:
        student = StudentModel.from_run(run, RUNS / run_name / "checkpoints" / f"checkpoint_epoch_{epoch}.pt", device="cpu")
        tf = student.teacher_forced(example)
        s_logits, t_logits, targets = tf["prediction_logits"], tf["teacher_logits"], tf["target_indices"]
        s_logp = F.log_softmax(s_logits, -1); t_p = F.softmax(t_logits, -1)
        kl = F.kl_div(s_logp, t_p, reduction="none").sum(-1)
        ce = F.cross_entropy(s_logits, targets, reduction="none")
        tl = student.model.vocabulary["token_list"]
        rows = []
        for i in range(min(N, len(targets))):
            g = int(targets[i])
            t_top = torch.topk(t_p[i], 3); s_top = torch.topk(s_logp[i].exp(), 3)
            H = float(-(t_p[i] * torch.log(t_p[i].clamp_min(1e-12))).sum())
            rows.append({"token": tl[g], "teacher_p_gold": float(t_p[i, g]), "student_p_gold": float(s_logp[i, g].exp()),
                         "kl": float(kl[i]), "ce": float(ce[i]), "teacher_H": H,
                         "teacher_top": [(tl[int(j)], float(v)) for v, j in zip(t_top.values, t_top.indices)],
                         "student_top": [(tl[int(j)], float(v)) for v, j in zip(s_top.values, s_top.indices)]})
        key = f"{run_name}@epoch{epoch}"
        out["rows"][key] = rows
        print(key, "mean KL %.3f mean CE %.3f over all %d steps" % (float(kl.mean()), float(ce.mean()), len(targets)))
        out.setdefault("means", {})[key] = {"kl": float(kl.mean()), "ce": float(ce.mean()), "steps": len(targets)}
json.dump(out, open(r"C:/Users/Ors/AppData/Local/Temp/claude/D--preference-destillation/ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f/scratchpad/slideshow/pertoken.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
