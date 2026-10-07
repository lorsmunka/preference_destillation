import sys
sys.path.insert(0, r"D:\preference_destillation")
from library.shared.metrics.task import ExactAnswerMetric, ClassificationMetric, MathMetric
m = ExactAnswerMetric()
cases = ["Answer: 15;", " 15;", "24 - 9 = 15\nAnswer: 15;", "\nAnswer: 15;", " Answer: 15;",
         "24-9=15\nAnswer:15;", "Answer: 15; Answer: 16;", "Answer: 15", "foo;\nAnswer: 15;"]
for c in cases:
    print(repr(c), "->", repr(m._extract_answer(c)))
# A working-format student that skips the working but is right:
m2 = ExactAnswerMetric(); m2.update(["Answer", ":", "▁15", ";"], "24 - 9 = 15\nAnswer: 15;"); print("answer-only vs working gt:", m2.result().accuracy)
m3 = ExactAnswerMetric(); m3.update(["6", "▁*", "▁4", "▁=", "▁24", "\n", "Answer", ":", "▁15", ";"], "24 - 9 = 15\nAnswer: 15;"); print("different working, same answer:", m3.result().accuracy)
c = ClassificationMetric()
for text in ['"tone"', '5', '[1,2]', 'null']:
    try:
        c.update([text], '{"tone":"neutral","sentiment":"neutral","safety":"safe","toxicity":"respectful"}')
        print(text, "ok")
    except Exception as e:
        print(text, "CRASH", type(e).__name__, e)
