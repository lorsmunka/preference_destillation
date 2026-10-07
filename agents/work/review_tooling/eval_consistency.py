import sys, time, torch
sys.path.insert(0, r"D:\preference_destillation")
assert not torch.cuda.is_available()
from library import Analysis
name = sys.argv[1]
a = Analysis(name)
t = time.time()
r = a.evaluate(split="test", checkpoint="latest", k=20)
logged = a.run.final_eval
print("seconds", round(time.time() - t))
for field in ["teacher_forced_accuracy", "student_accuracy", "task_accuracy", "validity_rate", "topk_accuracy", "teacher_student_overlap",
              "mean_target_rank", "teacher_entropy", "student_entropy", "perplexity", "termination_rate", "mean_length_ratio"]:
    print(f"{field:26} evaluate={getattr(r, field)!s:22} logged={getattr(logged, field)}")
print("steps", r.steps, "logged", logged.total_steps, "examples", r.examples)
