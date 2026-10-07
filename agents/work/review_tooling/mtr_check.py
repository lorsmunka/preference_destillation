import sys, os
os.chdir(r"D:\preference_destillation"); sys.path.insert(0, r"D:\preference_destillation")
import math_tutor_report as m
data = {"human": m.human_sets(), "writing": m.writing(), "pruning": m.pruning(), "renumber": m.renumber_example(),
        "templateness": m.templateness()}
qa = m.methods_qa(data)
print("ok; sections:", {k: (v is not None) for k, v in data.items()}, "qa", len(qa))
