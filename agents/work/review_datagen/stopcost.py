import time
from transformers import AutoTokenizer
from transformers.generation.stopping_criteria import StopStringCriteria
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
for stops in ([";"], ["</solution>", "</explanation>", "<out_of_scope>"]):
    times = []
    for _ in range(4):
        t = time.time(); StopStringCriteria(tok, stops); times.append(round(time.time() - t, 3))
    print(stops, "construction seconds (first, then cached):", times)
