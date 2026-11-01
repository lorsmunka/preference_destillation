"""Per-token teacher entropy for the thesis 2.2 examples -> data/12_token_entropy_examples.json (figure 2.1).
Entropy: softmax over the stored reduced-vocabulary logits at T=1, nats."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\preference_destillation")
from library.shared.batch_files import batch_paths, bfloat16_bits_to_float32

ROOT = r"D:\preference_destillation"
OUTPUT = r"D:\preference_destillation\thesis_work\grafikonok\data\12_token_entropy_examples.json"


def entropies(rows):
    rows = np.asarray(rows, dtype=np.float64)
    rows = rows - rows.max(axis=1, keepdims=True)
    probabilities = np.exp(rows)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return (-(probabilities * np.log(np.clip(probabilities, 1e-30, None))).sum(axis=1)).tolist()


def from_jsonl(path, line_number, must_contain):
    with open(path, encoding="utf-8") as file:
        record = json.loads(file.readlines()[line_number])
    assert must_contain in record["sentence"], record["sentence"][:80]
    return {"tokens": [step["token"] for step in record["steps"]],
            "entropy": entropies([step["logits"] for step in record["steps"]])}


def corpus_text(path, predicate):
    with open(path, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if predicate(record):
                return record["text"]
    raise SystemExit(f"no corpus record in {path}")


def from_npz(folder, wanted_texts):
    """Find each wanted input text among the batch records; returns {text: panel}."""
    found = {}
    for number, path in enumerate(batch_paths(folder)):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
            hits = [index for index, sentence in enumerate(meta["sentences"]) if sentence in wanted_texts]
            if not hits:
                continue
            logits = bfloat16_bits_to_float32(data["logits_bf16"])
            offsets = data["step_offsets"]
        for index in hits:
            rows = logits[int(offsets[index]):int(offsets[index + 1])]
            found[meta["sentences"][index]] = {"tokens": meta["tokens"][index], "entropy": entropies(rows)}
        print(f"  found {len(found)}/{len(wanted_texts)} after {number + 1} files", flush=True)
        if len(found) == len(wanted_texts):
            break
    missing = [text[:60] for text in wanted_texts if text not in found]
    if missing:
        raise SystemExit(f"not found: {missing}")
    return found


panels = {}
panels["reddit"] = from_jsonl(ROOT + r"\batches\reddit_comment_sentiment\google_gemma-3-4b-it\batch_15400.jsonl", 13, "You only need 1 friend")
panels["math_v1"] = from_jsonl(ROOT + r"\batches\math_word_problem\google_gemma-3-4b-it\batch_1.jsonl", 1, "Olivia has 93 seeds")
panels["post_generation"] = from_jsonl(ROOT + r"\batches\post_generation\google_gemma-3-4b-it\batch_1.jsonl", 2, "Main problem was they showed it")
print("part 1 panels done", flush=True)

tutor_corpus = ROOT + r"\library\domain\math_tutor\corpus_full_v1.jsonl"
tutor_solve = corpus_text(tutor_corpus, lambda r: r["item_id"] == "S08-mem-b06-0032" and r["turn_type"] == "solve" and r["turn"] == 0)
tutor_explain = corpus_text(tutor_corpus, lambda r: r["item_id"] == "S08-mem-b06-0032" and r["turn_type"] == "explain" and "16 and 5" in r["text"])
tutor = from_npz(ROOT + r"\batches\math_tutor_full_v1\google_gemma-4-12B-it", {tutor_solve, tutor_explain})
panels["tutor_solution"] = tutor[tutor_solve]
panels["tutor_explanation"] = tutor[tutor_explain]

code_corpus = ROOT + r"\library\domain\code_agent\corpus_v1.jsonl"
is_example = lambda r: (r["split"] == "test" and r["project"] == "todo" and r["truth"] == [["main.py", "ask_new_task"]]
                        and r["kind"] == "explain" and r["source"] == "name")
code_scope = corpus_text(code_corpus, lambda r: is_example(r) and r["turn_type"] == "scope")
code_explain = corpus_text(code_corpus, lambda r: is_example(r) and r["turn_type"] == "explain")
code = from_npz(ROOT + r"\batches\code_agent\google_gemma-4-12B-it", {code_scope, code_explain})
panels["code_scope"] = code[code_scope]
panels["code_explanation"] = code[code_explain]

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(panels, file, ensure_ascii=False)
for name, panel in panels.items():
    print(f"{name}: {len(panel['tokens'])} tokens, mean entropy {sum(panel['entropy']) / len(panel['entropy']):.3f}, "
          f"first tokens {''.join(panel['tokens'][:12])!r}")
