"""Gemma 4 12B itself, in bf16 on a rented card (never the local 4-bit copy), as the reference the
students are read against (chapter 9 plan #3, 2026-11-06):

  mass     How much of Gemma's probability lies inside each agent's reduced output vocabulary, at every
           position of its own replies (teacher-forced, the same prompt and target as the teacher data):
           500 test records per agent, drawn with random.Random(0). The students only ever see Gemma's
           distribution renormalised over that vocabulary, so the share outside it is what they lose.
  answers  Gemma's own first-turn solutions on the problems of experiments/math_generalization_check.py
           (the same 300 + 300 + 300 drawn with random.Random(0): test families, held-out shapes, SVAMP),
           with the same request the students get (greedy, 120 tokens, no re-ask), scored against the
           true answer. So the students' generalisation can be read as agreement with Gemma.
  report   (at home, CPU) The summaries into experiments/results/gemma_reference.csv, and the full-run math
           students' answers on the same problems scored as agreement with Gemma's answers.

    python experiments/gemma_reference.py mass       # on the card (~10-20 min)
    python experiments/gemma_reference.py answers    # on the card (~10 min)
    python experiments/gemma_reference.py report     # at home: zero arguments does this

Per-record results go to runs/gemma_reference/*.jsonl (appended; finished records are skipped on a rerun).
"""

import json
import random
import re
import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.agent.runner import Request  # noqa: E402
from library.domain.code_agent import CodeAgentDomain  # noqa: E402
from library.domain.math_tutor.conversation import Conversation  # noqa: E402
from library.domain.math_tutor.full import FullDomain  # noqa: E402
from library.domain.math_tutor.harness import SOLUTION_PREFIX  # noqa: E402
from library.shared.batch_files import vocabulary_fingerprint  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "runs" / "gemma_reference"
TEACHER = "google/gemma-4-12B-it"
AGENTS = {"math": FullDomain("v1"), "code": CodeAgentDomain("v1")}
BATCHES = {"math": REPO / "batches/math_tutor_full_v1/google_gemma-4-12B-it",
           "code": REPO / "batches/code_agent/google_gemma-4-12B-it"}
RECORDS_PER_AGENT = 500
SCORE_BATCH = {"math": 16, "code": 2}          # the coding agent's prompts hold a whole project (~7,000 tokens)
PER_SET = 300
ITEMS = REPO / "library/domain/math_tutor/items/items.jsonl"
SVAMP = REPO / "library/domain/math_tutor/benchmarks/real_test.jsonl"
ANSWER = re.compile(r"Answer:\s*\$?(-?[\d,]*\.?\d+)")
STUDENTS = [f"math-agent-{arm}-m-s{seed}" for arm in ("kl", "ce") for seed in (1, 2, 3)] + \
           [f"math-agent-kl-to-ce-m-s{seed}" for seed in (1, 2)]
RESULT_COLUMNS = ["row", "agent", "what", "value", "count", "note"]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def load_teacher(domain):
    from library.data_gen.model_handler import ModelHandler
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        raise SystemExit("Gemma is measured in bf16 only: run this on a card with >= 40 GB (never the local 4-bit copy).")
    return ModelHandler(TEACHER, domain, load_in_4bit=False, chat_template=True)


# ── mass: Gemma's probability inside each reduced output vocabulary ────────────────────────────────────

def test_records(agent: str, count: int):
    """The drawn test records, kept in runs/gemma_reference/records_<agent>.jsonl once drawn, so the card
    needs only that file and not the whole corpus (~200 MB each)."""
    path = OUTPUT / f"records_{agent}.jsonl"
    if path.exists():
        return read_jsonl(path)
    with open(AGENTS[agent].corpus_path, encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    tests = [record for record in records if record.get("split") == "test"]
    drawn = random.Random(0).sample(tests, min(count, len(tests)))
    append_jsonl(path, drawn)
    return drawn


def inside_shares(handler, texts, targets):
    """For each (text, target): the share of Gemma's probability inside the reduced vocabulary at every
    target position, and whether its top word is inside. The same prompt, padding and positions as
    ModelHandler.score_training_examples, with the full softmax instead of the reduced logits."""
    prompt_ids = [handler.prompt_token_ids(handler.build_prompt(text)) for text in texts]
    target_ids = [handler.tokenizer(target, add_special_tokens=False)["input_ids"] for target in targets]
    sequences = [prompt + target for prompt, target in zip(prompt_ids, target_ids)]
    length = max(len(sequence) for sequence in sequences)
    keep = max(len(target) for target in target_ids) + 1
    pad = handler.tokenizer.pad_token_id if handler.tokenizer.pad_token_id is not None else handler.tokenizer.eos_token_id
    device = next(handler.model.parameters()).device
    input_ids = torch.tensor([[pad] * (length - len(s)) + s for s in sequences], device=device)
    attention_mask = torch.tensor([[0] * (length - len(s)) + [1] * len(s) for s in sequences], device=device)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    token_list = handler.vocabulary["token_list"]
    reduced_ids = torch.tensor([handler.vocabulary["token_to_id"][token] for token in token_list], device=device)
    with torch.no_grad():
        logits = handler.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               logits_to_keep=keep).logits
        inside = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        inside[reduced_ids] = True
        results = []
        for row, ids in enumerate(target_ids):
            offset = keep - len(ids) - 1
            window = logits[row, offset:offset + len(ids)].float()
            probabilities = torch.softmax(window, dim=-1)
            shares = probabilities[:, inside].sum(-1)
            top_inside = inside[window.argmax(-1)]
            target_inside = inside[torch.tensor(ids, device=device)]
            results.append({"shares": [round(value, 6) for value in shares.tolist()],
                            "top_inside": top_inside.tolist(), "target_inside": target_inside.tolist()})
    return results


def mass() -> None:
    for agent, domain in AGENTS.items():
        path = OUTPUT / f"mass_{agent}.jsonl"
        done = {row["item_id"] + "|" + row["turn_type"] for row in read_jsonl(path)}
        records = [record for record in test_records(agent, RECORDS_PER_AGENT)
                   if record["item_id"] + "|" + record["turn_type"] not in done]
        if not records:
            print(f"{agent}: done")
            continue
        handler = load_teacher(domain)
        info = json.loads((BATCHES[agent] / "info.json").read_text(encoding="utf-8"))
        fingerprint = vocabulary_fingerprint(handler.vocabulary["token_list"])
        if fingerprint != info["vocabulary_sha1"]:
            raise SystemExit(f"{agent}: the vocabulary here ({fingerprint}) is not the teacher data's ({info['vocabulary_sha1']})")
        print(f"{agent}: {len(records)} records, vocabulary {len(handler.vocabulary['token_list'])} tokens ({fingerprint[:8]})", flush=True)
        size = SCORE_BATCH[agent]
        for start in range(0, len(records), size):
            chunk = records[start:start + size]
            outcomes = inside_shares(handler, [record["text"] for record in chunk], [record["target"] for record in chunk])
            append_jsonl(path, [{"item_id": record["item_id"], "turn_type": record["turn_type"], **outcome}
                                for record, outcome in zip(chunk, outcomes)])
            print(f"  {start + len(chunk)}/{len(records)}", flush=True)
        del handler
        torch.cuda.empty_cache()


# ── answers: Gemma's own first-turn solutions on the generalisation sets ──────────────────────────────

def steps_of(shape):
    return shape.count(">") + 1


def generalisation_sets():
    """The problems of experiments/math_generalization_check.py, drawn the same way."""
    items = read_jsonl(ITEMS)
    sets = {
        "test": [(item["problem"], item["answers"][0], steps_of(item["problem_shape"])) for item in items if item["split"] == "test"],
        "heldout": [(item["problem"], item["answers"][0], steps_of(item["problem_shape"])) for item in items if item["split"] == "heldout_shape"],
        "svamp": [(item["problem"], item["answer"], item["steps"]) for item in read_jsonl(SVAMP)],
    }
    generator = random.Random(0)
    return {name: generator.sample(problems, min(PER_SET, len(problems))) for name, problems in sets.items()}


def answer_of(reply: str):
    found = ANSWER.search(reply)
    if not found:
        return None
    try:
        return float(found.group(1).replace(",", ""))
    except ValueError:
        return None


def is_correct(answer, truth) -> bool:
    return answer is not None and abs(answer - float(truth)) < 1e-6


def solve_requests(problems):
    return [Request("solution", Conversation().solve_context(problem), SOLUTION_PREFIX, stop=["</solution>"])
            for problem, _, _ in problems]


def answers() -> None:
    from library.agent.runner import TeacherModel
    path = OUTPUT / "answers_gemma.jsonl"
    if read_jsonl(path):
        print("Gemma's answers: done")
        return
    domain = AGENTS["math"]
    tutor = TeacherModel(load_teacher(domain), domain, batch=32)
    rows = []
    for name, problems in generalisation_sets().items():
        replies = tutor.generate(solve_requests(problems))
        for (problem, truth, steps), reply in zip(problems, replies):
            answer = answer_of(reply)
            rows.append({"set": name, "problem": problem, "truth": truth, "steps": steps, "reply": reply,
                         "answer": answer, "correct": is_correct(answer, truth)})
        print(f"{name}: Gemma right {mean(row['correct'] for row in rows if row['set'] == name):.3f}", flush=True)
    append_jsonl(path, rows)


# ── report (at home) ──────────────────────────────────────────────────────────────────────────────────

def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


def mass_rows():
    rows = []
    for agent in AGENTS:
        records = read_jsonl(OUTPUT / f"mass_{agent}.jsonl")
        if not records:
            continue
        shares = [share for record in records for share in record["shares"]]
        tops = [inside for record in records for inside in record["top_inside"]]
        targets = [inside for record in records for inside in record["target_inside"]]
        note = f"{len(records)} test records, {len(shares)} positions"
        values = {"mean share inside": mean(shares), "median share inside": percentile(shares, 0.5),
                  "5th percentile share inside": percentile(shares, 0.05),
                  "positions with < 90% inside": mean(share < 0.9 for share in shares),
                  "record mean share, lowest": min(mean(record["shares"]) for record in records),
                  "top word inside": mean(tops), "written word inside": mean(targets)}
        for turn_type in sorted({record["turn_type"] for record in records}):
            subset = [share for record in records if record["turn_type"] == turn_type for share in record["shares"]]
            values[f"mean share inside, {turn_type}"] = mean(subset)
        rows += [{"row": f"mass-{agent}-{what}", "agent": agent, "what": what, "value": round(value, 5),
                  "count": len(shares), "note": note} for what, value in values.items()]
    return rows


def student_answers(gemma_rows):
    """Each full-run math student's answers on the same problems (CPU, cached per student)."""
    from library.domain.math_tutor.demo import load_student
    path = OUTPUT / "answers_students.jsonl"
    done = {row["student"] for row in read_jsonl(path)}
    problems = [(row["problem"], row["truth"], row["steps"]) for row in gemma_rows]
    for name in STUDENTS:
        if name in done:
            continue
        _, model = load_student(name)
        replies = model.generate(solve_requests(problems))
        append_jsonl(path, [{"student": name, "set": row["set"], "problem": row["problem"], "answer": answer_of(reply)}
                            for row, reply in zip(gemma_rows, replies)])
        print(f"{name}: answered {len(replies)}", flush=True)
    return read_jsonl(path)


def answer_rows():
    gemma_rows = read_jsonl(OUTPUT / "answers_gemma.jsonl")
    if not gemma_rows:
        return []
    rows = []
    for name in ("test", "heldout", "svamp"):
        subset = [row for row in gemma_rows if row["set"] == name]
        rows.append({"row": f"gemma-right-{name}", "agent": "math", "what": f"Gemma right vs the true answer, {name}",
                     "value": round(mean(row["correct"] for row in subset), 4), "count": len(subset),
                     "note": "first attempt, greedy, 120 tokens, no re-ask (the students' request)"})
    gemma_answer = {(row["set"], row["problem"]): row["answer"] for row in gemma_rows}
    gemma_correct = {(row["set"], row["problem"]): row["correct"] for row in gemma_rows}
    truth = {(row["set"], row["problem"]): row["truth"] for row in gemma_rows}
    for student in STUDENTS:
        answers_of_student = [row for row in student_answers(gemma_rows) if row["student"] == student]
        for name in ("test", "heldout", "svamp"):
            subset = [row for row in answers_of_student if row["set"] == name]
            keys = [(row["set"], row["problem"]) for row in subset]
            agree = mean(row["answer"] is not None and gemma_answer[key] is not None
                         and abs(row["answer"] - gemma_answer[key]) < 1e-6 for row, key in zip(subset, keys))
            right = mean(is_correct(row["answer"], truth[key]) for row, key in zip(subset, keys))
            where_gemma_right = [is_correct(row["answer"], truth[key]) for row, key in zip(subset, keys) if gemma_correct[key]]
            for what, value, count in (("agrees with Gemma's answer", agree, len(subset)),
                                       ("right vs the true answer", right, len(subset)),
                                       ("right where Gemma is right", mean(where_gemma_right), len(where_gemma_right))):
                rows.append({"row": f"{student}-{name}-{what}", "agent": "math", "what": f"{student}: {what}, {name}",
                             "value": round(value, 4), "count": count, "note": "same request as Gemma's"})
    return rows


def report() -> None:
    from _common import write_rows
    rows = mass_rows() + answer_rows()
    if not rows:
        raise SystemExit("nothing to report yet: run mass / answers on the card first")
    write_rows("gemma_reference.csv", RESULT_COLUMNS, rows)
    for row in rows:
        print(f"{row['row']:70s} {row['value']}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"mass": mass, "answers": answers, "report": report}[step]()
