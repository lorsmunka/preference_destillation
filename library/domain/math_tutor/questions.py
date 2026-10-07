"""The pupil's follow-up messages for the math tutor, from the template bank (questions/bank/*.jsonl,
library/agent/question_templates.py). This module turns one turn of a conversation into foci: the
things a message can be about, each with the slot values and features a template may use.

    step        one calculation line of the shown solution: "number of bags = 63 / 7 = 9"
    fact        one fact line: "bottles per bag = 7"
    answer      the solution as a whole
    distractor  a number in the problem that the solution rightly does not use
    mistake     a plausible wrong answer, computed: wrong operation, used the distractor,
                skipped the last step, or a slip (pushback: the tutor must hold its ground)
    what_if     a number of this turn changed: the problem's code is run again, so the new
                answer (and the facts-first solution, carried over) is known
    none        nothing in particular (off-topic messages, "I still don't get it")

The slots come from the solution the pupil sees, so they work on a gold label and on the
tutor's own live solution alike.

    python -m library.domain.math_tutor.questions check questions/bank/why.jsonl   # writers' check
    python -m library.domain.math_tutor.questions stats                            # the whole bank
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from library.agent.question_templates import Focus, QuestionGenerator, load_bank, template_problem
from .annotations import turn_executions
from .checks import REMAINDER, check_against_trace, parse
from .functions import NotAllowed, execute
from .items import REMAINDER_SLICE, code_literals, constants_allowed, digit_numbers, text_numbers
from .labels import computed_values, gold_calculations, transfer
from .renumber import draw, replace_in_code, replace_in_text

PACKAGE = Path(__file__).resolve().parent
BANK_DIR = PACKAGE / "questions" / "bank"
SAMPLE_LABELS = PACKAGE / "questions" / "sample_labels.jsonl"   # verified facts-first labels of ~90 items
EXCLUDED = PACKAGE / "questions" / "excluded.jsonl"   # templates taken out after a test, with the reason (appended)

KINDS = {   # kind -> the route the tutor should take
    "why": "explanation", "meaning": "explanation", "where_from": "explanation", "check": "explanation",
    "another_way": "explanation", "lost": "explanation", "shortcut": "explanation",
    "smaller_numbers": "explanation", "wrong_operation": "explanation", "distractor": "explanation",
    "pushback": "explanation", "follow": "explanation", "what_if": "solution", "off_topic": "out_of_scope",
}
KIND_NOTES = {
    "why": "why the tutor did a step (this operation, these numbers, this order)",
    "meaning": "what a number or a step stands for in the story; often as a guess in the story's words, asking if it is right",
    "where_from": "where a number came from (the problem, an earlier answer, a word like 'pair')",
    "check": "whether a line or the answer is really right, or how the tutor knows",
    "another_way": "whether a step or the whole problem could be done another way",
    "lost": "the pupil does not understand a step and asks for it more simply",
    "shortcut": "whether there is a quicker way",
    "smaller_numbers": "to see a step again with smaller, easier numbers",
    "wrong_operation": "why the tutor did not use another operation here (why not take away, why not multiply)",
    "distractor": "about a number in the problem that the solution does not use",
    "pushback": "the pupil says the tutor is wrong and gives their own (wrong) answer; the tutor must hold firm",
    "follow": "a follow-up to the tutor's last explanation: still lost, why, say it simpler, an example, so does that mean...",
    "what_if": "what the answer would be if a number were different, or the pupil gives new numbers to work with "
               "(a new calculation: the tutor solves again)",
    "off_topic": "not about this maths at all (other subjects, personal questions, jokes, games, rude); the harness answers",
}

FOCUS_SLOTS = {
    "step": {
        "step_words": "what the line computes, in the solution's words: 'number of bags'",
        "calculation": "the calculation as written: '63 / 7'",
        "result": "its result: 9",
        "ordinal": "its position: 'first', 'second', 'third', 'fourth'",
        "a": "first number of a one-operation step: 63", "b": "second number: 7",
        "sign": "the sign: '+', '-', '*' or '/'",
        "verb": "'add', 'take away', 'multiply' or 'divide'",
        "verb_past": "'added', 'took away', 'multiplied' or 'divided'",
        "sign_word": "'plus', 'minus', 'times' or 'divided by'",
        "operation_phrase": "the operation with its numbers: 'divide 63 by 7', 'take 9 away from 54', 'add 54 and 9'",
        "operation_phrase_past": "'divided 63 by 7', 'took 9 away from 54', 'added 54 and 9'",
        "other_verb": "an operation that would be wrong here: 'multiply'",
        "other_phrase": "the wrong operation with the same numbers: 'multiply 63 by 7'",
        "a_words": "the words for a, from the solution: 'new total'", "b_words": "the words for b: 'bottles per bag'",
        "quotient": "a remainder step's whole part: 5", "remainder": "what is left over: 1",
    },
    "fact": {"number": "the number: 7", "words": "what it is, in the solution's words: 'bottles per bag'"},
    "answer": {"answer": "the final answer: 9", "answer_words": "what the answer is: 'number of bags'",
               "step_count": "how many calculation lines: 2"},
    "distractor": {"distractor": "the unused number: 4",
                   "distractor_sentence": "the sentence of the problem it is in: 'Hayrides cost $4 each.'"},
    "mistake": {"wrong_answer": "the pupil's wrong answer: 45", "answer": "the tutor's answer: 63",
                "wrong_verb": "(wrong_operation) the operation the pupil used: 'take away'",
                "wrong_phrase": "(wrong_operation) it with the numbers: 'take 9 away from 54'",
                "distractor": "(used_distractor) the unused number the pupil used: 4",
                "skipped_words": "(skipped_step) the step the pupil left out: 'number of bags'"},
    "what_if": {"old": "the number in this turn's task: 9", "new": "the changed number: 12",
                "old_words": "the solution's words for it: 'bottles Ines added'"},
    "none": {},
}
FEATURES = {
    # step
    "add": "a one-operation addition step", "subtract": "a subtraction step", "multiply": "a multiplication step",
    "divide": "a division step", "multi_operation": "a step with more than one operation",
    "remainder": "a division that leaves a remainder", "rounded_up": "a remainder rounded up (16 / 3 -> 6 boxes)",
    "uses_previous_step": "the step uses the result of an earlier line",
    "uses_earlier_answer": "the step uses the answer of an earlier task",
    "first_step": "the first calculation", "last_step": "the last calculation", "only_step": "the only calculation",
    "has_words": "the solution names both numbers of the step (a_words and b_words exist)",
    # fact
    "from_problem": "the number is written in the problem", "from_earlier_answer": "the number is an earlier answer",
    "implied_by_word": "the number comes from a word: a pair = 2, a week = 7, a dozen = 12",
    # answer
    "one_step": "the solution has one calculation", "multi_step": "two or more calculations",
    # mistake
    "wrong_operation": "the pupil used another operation", "used_distractor": "the pupil used the unused number",
    "skipped_step": "the pupil stopped one step early", "slip": "a small arithmetic slip",
    # the whole conversation
    "follow_up_task": "this turn is a follow-up task, building on an earlier answer",
    "has_distractor": "the problem has an unused number", "money": "the story is about money",
    "time": "the story is about time (minutes, hours, days, weeks)",
    "join": "story type: something is added to an amount", "separate": "story type: something is taken away",
    "combine": "story type: two parts make a whole", "compare": "story type: how many more / fewer",
    "equal_groups": "story type: equal groups (multiplication)", "sharing": "story type: sharing equally (division)",
    "grouping": "story type: how many groups of a size (division, can leave a remainder)",
    "times_as_many": "story type: times as many", "rates": "story type: rates and prices",
    "mixed": "story type: several different steps",
}
SLICE_FEATURES = {"S01": {"join"}, "S02": {"separate"}, "S03": {"combine"}, "S04": {"compare"},
                  "S05": {"equal_groups"}, "S06": {"sharing"}, "S07": {"grouping"}, "S08": {"times_as_many"},
                  "S09": {"rates"}, "S10": {"mixed"}}
KIND_WEIGHTS = {
    "first": {"why": 10, "meaning": 8, "where_from": 7, "check": 6, "another_way": 5, "lost": 7, "shortcut": 4,
              "smaller_numbers": 5, "wrong_operation": 7, "distractor": 6, "pushback": 10, "what_if": 6, "off_topic": 4},
    "follow": {"follow": 25, "why": 8, "meaning": 6, "where_from": 8, "check": 4, "another_way": 3, "lost": 6,
               "shortcut": 2, "smaller_numbers": 5, "wrong_operation": 4, "distractor": 3, "pushback": 6,
               "what_if": 4, "off_topic": 3},
}

OPERATIONS = {
    "+": ("add", "added", "plus", "add {a} and {b}", "added {a} and {b}"),
    "-": ("take away", "took away", "minus", "take {b} away from {a}", "took {b} away from {a}"),
    "*": ("multiply", "multiplied", "times", "multiply {a} by {b}", "multiplied {a} by {b}"),
    "/": ("divide", "divided", "divided by", "divide {a} by {b}", "divided {a} by {b}"),
}
FEATURE_OF_SIGN = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
WRONG_SIGN = {"+": "-", "-": "+", "*": "+", "/": "*"}
ONE_OPERATION = re.compile(r"^\s*(\d+)\s*([-+*/])\s*(\d+)\s*$")
REMAINDER_OF = re.compile(r"^\s*remainder of (\d+) / (\d+)\s*$")
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]
MONEY = re.compile(r"\$|\b(dollars?|cents?|pence|pounds?|coins?|price|costs?|pay|paid|money)\b", re.IGNORECASE)
TIME = re.compile(r"\b(minutes?|hours?|days?|weeks?|months?|seconds?)\b", re.IGNORECASE)


def tasks_of(item: Dict) -> List[str]:
    return [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]]


def sources_of(item: Dict) -> List[str]:
    return [item["solve"]] + [follow_up["solve"] for follow_up in item["follow_ups"]]


def apply(a: int, sign: str, b: int) -> Optional[int]:
    if sign == "+":
        return a + b
    if sign == "-":
        return a - b if a >= b else None
    if sign == "*":
        return a * b
    return a // b if b and a % b == 0 else None


def operation_slots(a: int, sign: str, b: int, prefix: str = "") -> Dict[str, object]:
    verb, past, word, phrase, phrase_past = OPERATIONS[sign]
    if sign == "-" and a < b:
        a, b = b, a
    return {f"{prefix}verb": verb, f"{prefix}verb_past": past, f"{prefix}sign_word": word,
            f"{prefix}phrase" if prefix else "operation_phrase": phrase.format(a=a, b=b),
            f"{prefix}phrase_past" if prefix else "operation_phrase_past": phrase_past.format(a=a, b=b)}


def conversation_features(item: Dict, turn: int) -> set:
    text = " ".join(tasks_of(item)[:turn + 1])
    features = set(SLICE_FEATURES.get(item["slice"], ()))
    if turn > 0:
        features.add("follow_up_task")
    if distractor_numbers(item):
        features.add("has_distractor")
    if MONEY.search(text):
        features.add("money")
    if TIME.search(text):
        features.add("time")
    return features


def distractor_numbers(item: Dict) -> List[int]:
    unused = Counter(n for n in digit_numbers(item["problem"]) if n > 1) - Counter(code_literals(item["solve"]))
    return sorted(unused)


def sentence_with(text: str, number: int) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(rf"(?<!\d){number}(?!\d)", sentence):
            return sentence.strip()
    return ""


def solution_foci(item: Dict, turn: int, block: str, random_generator: random.Random) -> List[Focus]:
    """Every focus of one shown solution (a gold label or the tutor's own block)."""
    shared = conversation_features(item, turn)
    lines = parse(block)
    tasks = tasks_of(item)
    written = set(text_numbers(" ".join(tasks[:turn + 1])))
    implied = constants_allowed(" ".join(tasks[:turn + 1]))
    earlier_answers = set(item["answers"][:turn])
    words_of: Dict[int, str] = {}
    foci = [Focus("none", key="none", features=set(shared))]

    facts = [line for line in lines if line.kind == "fact"]
    for index, line in enumerate(facts):
        features = set(shared)
        if line.number in earlier_answers and line.number not in set(text_numbers(tasks[turn])):
            features.add("from_earlier_answer")
        elif line.number in written:
            features.add("from_problem")
        elif line.number in implied:
            features.add("implied_by_word")
        words_of.setdefault(line.number, line.label)
        foci.append(Focus("fact", {"number": line.number, "words": line.label}, features, key=f"fact{index}"))

    calculations = [line for line in lines if line.kind == "calculation"]
    result_keys: Dict[int, str] = {}
    for index, line in enumerate(calculations):
        expression = line.text.split("=", 1)[1].rsplit("=", 1)[0].strip() if line.text.count("=") >= 2 else ""
        remainder_line = REMAINDER.match(f"{line.text.split('=', 1)[1].strip()}") if "remainder" in line.text else None
        slots: Dict[str, object] = {"step_words": line.label, "result": line.number,
                                    "ordinal": ORDINALS[min(index, len(ORDINALS) - 1)]}
        features = set(shared)
        operands: List[int] = []
        single = ONE_OPERATION.match(expression)
        remainder_of = REMAINDER_OF.match(expression)
        if remainder_line:   # "16 / 3 = 5 remainder 1, rounded up = 6": the expression splits oddly, so read it whole
            a, b = int(remainder_line.group("a")), int(remainder_line.group("b"))
            slots.update(calculation=f"{a} / {b}", a=a, b=b, sign="/", quotient=int(remainder_line.group("q")),
                         remainder=int(remainder_line.group("r")), **operation_slots(a, "/", b))
            features |= {"divide", "remainder"} | ({"rounded_up"} if remainder_line.group("up") else set())
            operands = [a, b]
        elif remainder_of:
            a, b = int(remainder_of.group(1)), int(remainder_of.group(2))
            slots.update(calculation=expression, a=a, b=b, sign="/", remainder=line.number, **operation_slots(a, "/", b))
            features |= {"divide", "remainder"}
            operands = [a, b]
        elif single:
            a, sign, b = int(single.group(1)), single.group(2), int(single.group(3))
            slots.update(calculation=expression, a=a, b=b, sign=sign, **operation_slots(a, sign, b))
            wrong = WRONG_SIGN[sign]
            if apply(max(a, b) if wrong == "-" else a, wrong, min(a, b) if wrong == "-" else b) is not None:
                other = operation_slots(a, wrong, b, prefix="other_")
                slots.update(other_verb=other["other_verb"], other_phrase=other["other_phrase"])
            features.add(FEATURE_OF_SIGN[sign])
            operands = [a, b]
        else:
            slots["calculation"] = expression
            features.add("multi_operation")
            operands = [int(number) for number in re.findall(r"\d+", expression)]
        if "a" in slots and slots["a"] in words_of and slots["b"] in words_of:
            slots.update(a_words=words_of[slots["a"]], b_words=words_of[slots["b"]])
            features.add("has_words")
        links = []
        for number in operands:
            if number in result_keys:
                features.add("uses_previous_step")
                links.append(result_keys[number])
            if number in earlier_answers and number not in written:
                features.add("uses_earlier_answer")
            links += [f"fact{i}" for i, fact in enumerate(facts) if fact.number == number]
        if index == 0:
            features.add("first_step")
        if index == len(calculations) - 1:
            features.add("last_step")
        if len(calculations) == 1:
            features.add("only_step")
        key = f"step{index}"
        foci.append(Focus("step", {k: v for k, v in slots.items() if v is not None and v != ""}, features, key=key, links=links))
        if line.number is not None:
            result_keys.setdefault(line.number, key)
            words_of.setdefault(line.number, line.label)
    for focus in foci:   # a step also links forward, to the steps that use its result
        for key in focus.links:
            target = next((other for other in foci if other.key == key), None)
            if target is not None and focus.key not in target.links:
                target.links.append(focus.key)

    answer_line = lines[-1] if lines and lines[-1].kind == "answer" else None
    answer = answer_line.number if answer_line else None
    if answer is not None:
        answer_slots = {"answer": answer, "step_count": len(calculations)}
        if calculations:
            answer_slots["answer_words"] = calculations[-1].label
        foci.append(Focus("answer", answer_slots, set(shared) | {"one_step" if len(calculations) <= 1 else "multi_step"},
                          key="answer", links=[f"step{len(calculations) - 1}"] if calculations else []))
        foci += mistake_foci(calculations, answer, distractor_numbers(item), shared, random_generator)
    shown_numbers = {int(number) for line in lines for number in re.findall(r"\d+", line.text)}
    for index, number in enumerate(n for n in distractor_numbers(item) if n not in shown_numbers):   # still unused in this turn
        foci.append(Focus("distractor", {"distractor": number, "distractor_sentence": sentence_with(item["problem"], number)},
                          set(shared), key=f"distractor{index}"))
    return foci


def mistake_foci(calculations, answer: int, distractors: List[int], shared: set,
                 random_generator: random.Random) -> List[Focus]:
    """Plausible wrong answers a pupil might push back with."""
    foci = []
    last = calculations[-1] if calculations else None
    single = ONE_OPERATION.match(last.text.split("=", 1)[1].rsplit("=", 1)[0]) if last and last.text.count("=") >= 2 else None
    if single:
        a, sign, b = int(single.group(1)), single.group(2), int(single.group(3))
        wrong = WRONG_SIGN[sign]
        first, second = (max(a, b), min(a, b)) if wrong == "-" else (a, b)
        value = apply(first, wrong, second)
        if value is not None and value != answer and value <= 999:
            other = operation_slots(first, wrong, second, prefix="wrong_")
            foci.append(Focus("mistake", {"wrong_answer": value, "answer": answer, "wrong_verb": other["wrong_verb"],
                                          "wrong_phrase": other["wrong_phrase"]},
                              set(shared) | {"wrong_operation"}, key="mistake_operation", links=["answer"]))
    for distractor in distractors:
        value = answer + distractor if answer + distractor <= 999 else answer - distractor
        foci.append(Focus("mistake", {"wrong_answer": value, "answer": answer, "distractor": distractor},
                          set(shared) | {"used_distractor"}, key="mistake_distractor", links=["answer"]))
        break
    if len(calculations) >= 2 and calculations[-2].number not in (None, answer):
        foci.append(Focus("mistake", {"wrong_answer": calculations[-2].number, "answer": answer,
                                      "skipped_words": calculations[-1].label},
                          set(shared) | {"skipped_step"}, key="mistake_skipped", links=["answer"]))
    slip = answer + random_generator.choice([-10, -1, 1, 10])
    if slip >= 0:
        foci.append(Focus("mistake", {"wrong_answer": slip, "answer": answer}, set(shared) | {"slip"},
                          key="mistake_slip", links=["answer"]))
    return foci


def what_if_foci(item: Dict, turn: int, block: str, random_generator: random.Random, limit: int = 2) -> List[Focus]:
    """Change one number of this turn's task, run the code again: the new answer, and the shown
    solution carried over to the new number (kept only if it passes the label check)."""
    tasks, sources = tasks_of(item), sources_of(item)
    text, source = tasks[turn], sources[turn]
    implied = constants_allowed(text)
    candidates = sorted({value for value in text_numbers(text) if value > 1 and value in code_literals(source)
                         and value not in implied})
    random_generator.shuffle(candidates)
    executions = turn_executions(item)
    state = executions[turn - 1].result if turn > 0 else None
    words_of = {line.number: line.label for line in parse(block) if line.kind == "fact"}
    foci = []
    for old in candidates:
        for _ in range(4):
            new = draw(old, set(text_numbers(text)), random_generator)
            if new is None:
                break
            try:
                execution = execute(replace_in_code(source, {old: new}), state, item["slice"] == REMAINDER_SLICE, trace=True)
            except NotAllowed:
                continue
            answer = execution.result["answer"]
            if answer == item["answers"][turn]:
                continue
            changed = dict(item)
            if turn == 0:
                changed["problem"], changed["solve"] = replace_in_text(text, {old: new}), replace_in_code(source, {old: new})
            else:
                changed["follow_ups"] = [dict(f) for f in item["follow_ups"]]
                changed["follow_ups"][turn - 1] = {"question": replace_in_text(text, {old: new}),
                                                   "solve": replace_in_code(source, {old: new})}
            changed_executions = executions[:turn] + [execution]
            gold = transfer(block, {old: new}, gold_calculations(changed, turn, changed_executions), answer)
            context = " ".join(tasks[:turn + 1] + [str(n) for n in item["answers"][:turn]] + [str(new)])
            if gold and check_against_trace(gold, computed_values(changed_executions, turn), answer, context):
                gold = ""
            slots = {"old": old, "new": new}
            if old in words_of:
                slots["old_words"] = words_of[old]
            foci.append(Focus("what_if", slots, conversation_features(item, turn), key=f"what_if{old}",
                              data={"answer": answer, "gold": gold}))
            break
        if len(foci) >= limit:
            break
    return foci


def bank_files() -> List[Path]:
    return sorted(BANK_DIR.glob("*.jsonl"))


def excluded_texts() -> List[str]:
    if not EXCLUDED.exists():
        return []
    return [json.loads(line)["text"] for line in open(EXCLUDED, encoding="utf-8") if line.strip()]


def generator(paths: Optional[List[Path]] = None) -> QuestionGenerator:
    return QuestionGenerator(load_bank(paths or bank_files(), KINDS, excluded_texts()), KIND_WEIGHTS["first"])


def kind_weights(depth: int) -> Dict[str, float]:
    return KIND_WEIGHTS["first" if depth == 0 else "follow"]


# --- the writers' check ----------------------------------------------------------------------

def code_block(item: Dict, turn: int, executions) -> str:
    """A facts-first-like block straight from the code (names as words): for the writers' check
    on story types the verified sample does not cover. Never training data."""
    from .annotations import calculation, label
    steps = executions[turn].steps
    facts = [f"{label(step.name)} = {step.value}" for step in steps if not step.computes]
    calculations = [f"{label(step.name)} = {calculation(step, item)}" for step in steps if step.computes]
    return "\n".join(["<solution>"] + facts + calculations + [f"Answer: {executions[turn].result['answer']}", "</solution>"])


def sample_conversations(per_slice: int = 12) -> List[tuple]:
    """(item, turn, block): every verified label in the sample set, plus a few items of every
    story type with a block made from the code."""
    labels = {}
    for line in open(SAMPLE_LABELS, encoding="utf-8"):
        record = json.loads(line)
        labels[(record["item_id"], record["turn"])] = record["block"]
    ids = {item_id for item_id, _ in labels}
    items = [json.loads(line) for line in open(PACKAGE / "items" / "items.jsonl", encoding="utf-8")]
    conversations = [(item, turn, labels[(item["id"], turn)]) for item in items if item["id"] in ids
                     for turn in range(1 + len(item["follow_ups"])) if (item["id"], turn) in labels]
    random_generator = random.Random(0)
    for slice_name in sorted(SLICE_FEATURES):
        written = [item for item in items if item["slice"] == slice_name and item["id"] not in ids
                   and not str(item.get("source", "")).startswith("reword:")]
        for item in random_generator.sample(written, min(per_slice, len(written))):
            executions = turn_executions(item)
            conversations += [(item, turn, code_block(item, turn, executions)) for turn in range(len(executions))]
    return conversations


def check_file(path: Path, examples: int = 2) -> None:
    """Every problem in one bank file, then each template filled on real sample conversations."""
    focus_slots = {focus: set(slots) for focus, slots in FOCUS_SLOTS.items()}
    lines = [line for line in open(path, encoding="utf-8") if line.strip()]
    problems = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {number}: not JSON ({error})")
            continue
        problem = template_problem(record, KINDS, focus_slots, set(FEATURES))
        if problem is None and re.search(r"\d", record["text"]):
            problem = "digits in the text: every number must come from a slot"
        if problem:
            problems.append(f"line {number}: {problem}: {line.strip()[:120]}")
    print(f"{path.name}: {len(lines)} lines, {len(problems)} problems")
    for problem in problems:
        print("  " + problem)
    templates = load_bank([path], KINDS)
    random_generator = random.Random(0)
    conversations = sample_conversations()
    foci_by_conversation = []
    for item, turn, block in conversations:
        last = turn == len(item["follow_ups"])
        foci = solution_foci(item, turn, block, random_generator)
        if last:
            foci += what_if_foci(item, turn, block, random_generator)
        foci_by_conversation.append(foci)
    unused = 0
    for template in templates:
        fills = []
        applies_to = 0
        for foci in foci_by_conversation:
            matching = [focus for focus in foci if template.applies(focus, 1 if template.depth == "follow" else 0)]
            if matching:
                applies_to += 1
                if len(fills) < examples:
                    fills.append(template.fill(random_generator.choice(matching)))
        unused += applies_to == 0
        share = f"{100 * applies_to / max(1, len(foci_by_conversation)):.0f}%"
        print(f"[{template.kind}/{template.focus}{'/' + ','.join(template.requires) if template.requires else ''}] "
              f"applies to {share} of {len(foci_by_conversation)} sample turns")
        for fill in fills:
            print(f"    {fill}")
    print(f"{len(templates)} templates, {unused} apply to no sample turn (fine for rare niches, check the conditions)")


def stats() -> None:
    templates = load_bank(bank_files(), KINDS)
    print(f"{len(templates)} templates in {len(bank_files())} files")
    for (kind, focus), count in sorted(Counter((t.kind, t.focus) for t in templates).items()):
        print(f"  {kind:16} {focus:11} {count}")


def print_conversations(count: int) -> None:
    """The first `count` verified sample conversations, as the tutor model reads them."""
    from .conversation import Conversation
    shown = 0
    conversations = sample_conversations(per_slice=0)
    blocks = {(item["id"], turn): block for item, turn, block in conversations}
    for item, turn, block in conversations:
        if any((item["id"], earlier) not in blocks for earlier in range(turn)):
            continue
        conversation = Conversation()
        tasks = tasks_of(item)
        for earlier in range(turn + 1):
            body = blocks[(item["id"], earlier)].replace("<solution>", "").replace("</solution>", "").strip()
            conversation.add_solution(tasks[earlier], body)
        print(json.dumps({"item_id": item["id"], "turn": turn, "conversation": conversation.history(with_thread=False)},
                         ensure_ascii=False))
        shown += 1
        if shown >= count:
            break


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # stories have names like "café" (the Windows console is cp1252)
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        check_file(Path(sys.argv[2]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "conversations":
        print_conversations(int(sys.argv[2]) if len(sys.argv) >= 3 else 100)
    else:
        stats()
