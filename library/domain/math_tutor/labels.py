"""Label pass: the facts-first wording of every gold solution. Gemma gets the conversation
and the exact calculations of the item's solve() trace and writes the solution in words:
first each number it uses (what it is = number), then each calculation with a label, then
the answer. A block is kept only if checks.check_against_trace confirms it says exactly what
the function computed; "Gemma's words, the function's numbers".

Turns are labelled in waves (every first turn, then every second turn, ...) because a
follow-up's conversation shows the earlier turns' facts-first solutions. Renumbered copies
are not sent to Gemma: their original's labels are carried over with the copy's numbers and
checked again against the copy's own trace.

    python -m library.domain.math_tutor.labels      # GPU; resumable (done turns are skipped)

Writes items/labels/facts.jsonl (appended, one verified block per item turn) and
items/labels/facts_log.json (counts per wave, failures by reason).
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .annotations import calculation, turn_executions
from .checks import check_against_trace
from .conversation import Conversation

ITEMS_DIR = Path(__file__).resolve().parent / "items"
LABELS_PATH = ITEMS_DIR / "labels" / "facts.jsonl"
LOG_PATH = ITEMS_DIR / "labels" / "facts_log.json"
ATTEMPTS = 3          # greedy first, then two sampled retries
BATCH = 8             # long prompts: 8 keeps a 12 GB card off its memory limit

PROMPT = """You write the worked solution for a primary-school maths tutor.

Here is the tutoring conversation. The last block, <message>, is the task to solve:

{context}

These are the exact calculations that solve it, in this order:
{calculations}
The answer is {answer}.

Write the solution block:
- First, one line for each number you use, written as: what it is = number. Use words from the problem. A number from an earlier answer counts too. Write numbers without units.
- Then the calculations above, in the same order, each written as: what it is = calculation = result. You may split a long calculation into smaller steps.
- Then the line Answer: {answer}
- Then </solution>.
No numbering or bullet points: every line starts with its words.
Example:
<solution>
tetras bought later = 4
price of a tetra = 2
money left before = 13
cost of 4 more tetras = 4 * 2 = 8
money left = 13 - 8 = 5
Answer: 5
</solution>
Write only the block."""


def gold_calculations(item: Dict, turn: int, executions) -> List[str]:
    return [calculation(step, item) for step in executions[turn].steps if step.computes]


def computed_values(executions, turn: int) -> List[int]:
    """Every value any single operation of this turn's solve() produced (a line may split a
    multi-operation step into child-sized steps, or join two)."""
    return list(executions[turn].operation_values)


def read_labels() -> Dict[tuple, str]:
    labels = {}
    if LABELS_PATH.exists():
        for line in open(LABELS_PATH, encoding="utf-8"):
            record = json.loads(line)
            labels[(record["item_id"], record["turn"])] = record["block"]
    return labels


def solve_context(item: Dict, turn: int, labels: Dict[tuple, str]) -> Optional[str]:
    """The conversation for this turn, with earlier turns' verified facts-first solutions
    (None if an earlier turn has none yet)."""
    tasks = [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]]
    conversation = Conversation()
    for earlier in range(turn):
        block = labels.get((item["id"], earlier))
        if block is None:
            return None
        conversation.add_solution(tasks[earlier], block.replace("<solution>", "").replace("</solution>", "").strip())
    return conversation.solve_context(tasks[turn])


def label_wave(handler, items: List[Dict], turn: int, labels: Dict[tuple, str], log: Dict) -> None:
    """Label one turn index of every item that has it, appending verified blocks."""
    pending = []
    for item in items:
        if turn > len(item["follow_ups"]) or (item["id"], turn) in labels:
            continue
        context = solve_context(item, turn, labels)
        if context is None:
            log["blocked_by_earlier_turn"] += 1
            continue
        executions = turn_executions(item)
        pending.append((item, context, executions))
    failures = Counter()
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "a", encoding="utf-8") as out:
        for attempt in range(ATTEMPTS):
            still_failing = []
            for start in range(0, len(pending), BATCH):
                batch = pending[start:start + BATCH]
                prompts = [PROMPT.format(context=context, calculations="\n".join(gold_calculations(item, turn, executions)),
                                         answer=item["answers"][turn]) for item, context, executions in batch]
                texts = handler.generate_texts(prompts, max_new_tokens=220, temperature=0.0 if attempt == 0 else 0.7,
                                               stop_strings=["</solution>"], seed=attempt)
                for (item, context, executions), text in zip(batch, texts):
                    block = text[text.find("<solution>"):] if "<solution>" in text else "<solution>\n" + text.strip()
                    problem = check_against_trace(block, computed_values(executions, turn), item["answers"][turn], context)
                    if problem is None:
                        labels[(item["id"], turn)] = block
                        out.write(json.dumps({"item_id": item["id"], "turn": turn, "block": block, "attempt": attempt},
                                             ensure_ascii=False) + "\n")
                        out.flush()
                    else:
                        still_failing.append((item, context, executions))
                        examples = log.setdefault("failure_examples", [])
                        if len(examples) < 40:   # kept for diagnosis
                            examples.append({"item": item["id"], "turn": turn, "attempt": attempt, "problem": problem,
                                             "calculations": gold_calculations(item, turn, executions), "text": text})
                        if attempt == ATTEMPTS - 1:
                            failures[re.sub(r"\d+", "N", problem.split(":")[0])[:60]] += 1
            log["verified_by_attempt"][attempt] = log["verified_by_attempt"].get(attempt, 0) + len(pending) - len(still_failing)
            pending = still_failing
            if not pending:
                break
    log["failed"][turn] = dict(failures)


def run_label_pass(handler) -> Dict:
    """Every kept item (originals and rewordings), wave by wave."""
    items = [json.loads(line) for line in open(ITEMS_DIR / "items.jsonl", encoding="utf-8")]
    labels = read_labels()
    log = {"blocked_by_earlier_turn": 0, "verified_by_attempt": {}, "failed": {}}
    for turn in range(4):
        label_wave(handler, items, turn, labels, log)
        print(f"wave {turn}: {sum(1 for key in labels if key[1] == turn)} turns verified so far")
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


# --- renumbered copies: carry the original's labels over, then check again ---------------

def number_mapping(original: str, copy: str) -> Optional[Dict[int, int]]:
    """Old value -> new value, from the numbers in the original's and the copy's texts (same
    order, since renumbering only swaps numbers)."""
    old, new = re.findall(r"\d+", original), re.findall(r"\d+", copy)
    if len(old) != len(new):
        return None
    mapping = {}
    for a, b in zip(old, new):
        if mapping.get(int(a), int(b)) != int(b):
            return None
        mapping[int(a)] = int(b)
    return mapping


def transfer(original_block: str, mapping: Dict[int, int], copy_calculations: List[str], answer: int) -> str:
    """The original's facts with their numbers mapped, then the copy's own calculations under
    the original's calculation labels (labels' own numbers mapped too)."""
    swap = lambda text: re.sub(r"\d+", lambda match: str(mapping.get(int(match.group()), int(match.group()))), text)
    lines = [line for line in original_block.split("\n") if line and line not in ("<solution>", "</solution>")]
    facts = [swap(line) for line in lines if re.match(r"^[^=]+=\s*\d+\s*$", line) and not line.startswith("Answer")]
    labels = [line.split("=")[0].strip() for line in lines if line.count("=") >= 2]
    if len(labels) != len(copy_calculations):
        return ""
    calculations = [f"{swap(label)} = {calculation_text}" for label, calculation_text in zip(labels, copy_calculations)]
    return "\n".join(["<solution>"] + facts + calculations + [f"Answer: {answer}", "</solution>"])


def transfer_to_copies(labels: Dict[tuple, str]) -> Counter:
    """Carry every labelled original's blocks over to its renumbered copies (checked again)."""
    originals = {json.loads(line)["id"]: json.loads(line) for line in open(ITEMS_DIR / "items.jsonl", encoding="utf-8")}
    outcomes = Counter()
    with open(LABELS_PATH, "a", encoding="utf-8") as out:
        for line in open(ITEMS_DIR / "renumbered.jsonl", encoding="utf-8"):
            copy = json.loads(line)
            original = originals.get(copy["renumbered_from"])
            if original is None:
                outcomes["original not kept"] += 1
                continue
            texts = lambda item: " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]])
            mapping = number_mapping(texts(original), texts(copy))
            original_runs, copy_runs = turn_executions(original), turn_executions(copy)
            for old_run, new_run in zip(original_runs, copy_runs):   # earlier answers and results move too
                for old, new in zip([s.value for s in old_run.steps if s.computes], [s.value for s in new_run.steps if s.computes]):
                    if mapping is not None and mapping.setdefault(old, new) != new:
                        mapping = None
            if mapping is None:
                outcomes["numbers do not map one to one"] += 1
                continue
            for turn in range(len(copy_runs)):
                if (copy["id"], turn) in labels:
                    continue
                block = labels.get((original["id"], turn))
                if block is None:
                    outcomes["original turn not labelled"] += 1
                    break
                new_block = transfer(block, mapping, gold_calculations(copy, turn, copy_runs), copy["answers"][turn])
                context = solve_context(copy, turn, labels)
                problem = check_against_trace(new_block, computed_values(copy_runs, turn), copy["answers"][turn], context or "")
                if problem:
                    outcomes["transferred block fails the check"] += 1
                    break
                labels[(copy["id"], turn)] = new_block
                out.write(json.dumps({"item_id": copy["id"], "turn": turn, "block": new_block, "attempt": "transfer"},
                                     ensure_ascii=False) + "\n")
                outcomes["turns transferred"] += 1
    return outcomes


def main() -> None:
    from library.data_gen.model_handler import ModelHandler
    from .agentic import MathTutorAgenticDomain
    handler = ModelHandler("google/gemma-4-12B-it", MathTutorAgenticDomain("facts"), load_in_4bit=True, chat_template=True)
    log = run_label_pass(handler)
    log["copies"] = dict(transfer_to_copies(read_labels()))
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
