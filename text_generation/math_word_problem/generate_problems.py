import json
import random
import hashlib
import os
import sys

sys.path.append(os.path.dirname(__file__))

from word_problem_variables import NAMES, ITEMS, CONTAINERS, GROUPS
from operations import (
    ARITHMETIC_OPERATIONS, COMPARISON_OPERATIONS, LARGE_RANGE_ARITHMETIC,
    SCAFFOLDS, pick_distribution_bucket, generate_numbers,
)
from templates import (
    ADDITION_TEMPLATES, SUBTRACTION_TEMPLATES, MULTIPLICATION_TEMPLATES,
    DIVISION_TEMPLATES, MUL_ADD_TEMPLATES, MUL_SUB_TEMPLATES,
    ADD_MUL_TEMPLATES, ADD_SUB_TEMPLATES, SUB_MUL_TEMPLATES,
    DIV_ADD_TEMPLATES, CMP_TEMPLATES, CMP_SUB_TEMPLATES, CMP_MUL_TEMPLATES,
)

OPERATION_TEMPLATES = {
    "add": ADDITION_TEMPLATES, "sub": SUBTRACTION_TEMPLATES,
    "mul": MULTIPLICATION_TEMPLATES, "div": DIVISION_TEMPLATES,
    "mul_add": MUL_ADD_TEMPLATES, "mul_sub": MUL_SUB_TEMPLATES,
    "add_mul": ADD_MUL_TEMPLATES, "add_sub": ADD_SUB_TEMPLATES,
    "sub_mul": SUB_MUL_TEMPLATES, "div_add": DIV_ADD_TEMPLATES,
    "cmp": CMP_TEMPLATES, "cmp_sub": CMP_SUB_TEMPLATES, "cmp_mul": CMP_MUL_TEMPLATES,
}

TARGET_COUNT = 500_000


def pick_operation(is_comparison, range_high):
    if is_comparison:
        return random.choice(COMPARISON_OPERATIONS)
    if range_high >= 100:
        return random.choice(LARGE_RANGE_ARITHMETIC)
    return random.choice(ARITHMETIC_OPERATIONS)


def pick_names():
    name1, pronoun1, possessive1 = random.choice(NAMES)
    name2, _, _ = random.choice(NAMES)
    while name2 == name1:
        name2, _, _ = random.choice(NAMES)
    return name1, name2, pronoun1.capitalize(), possessive1.capitalize()


def fill_template(template, operation, range_low, range_high, use_round):
    name, name2, pronoun, possessive = pick_names()
    category = random.choice(list(ITEMS.keys()))
    items = random.choice(ITEMS[category])
    container = random.choice(CONTAINERS)
    group = random.choice(GROUPS)
    values = generate_numbers(operation, range_low, range_high, use_round)

    text = template.format(
        name=name, name2=name2, pronoun=pronoun, possessive=possessive,
        items=items, containers=container, container=container,
        groups=group, A=values.get("A", ""), B=values.get("B", ""),
        C=values.get("C", ""),
    )

    scaffold = SCAFFOLDS[operation]
    return f'Problem: "{text}"\n{scaffold}'


def generate_problems():
    seen = set()
    problems = []

    while len(problems) < TARGET_COUNT:
        _, range_low, range_high, use_round, is_comparison = pick_distribution_bucket()
        operation = pick_operation(is_comparison, range_high)
        template = random.choice(OPERATION_TEMPLATES[operation])
        problem_text = fill_template(template, operation, range_low, range_high, use_round)

        text_hash = hashlib.md5(problem_text.encode()).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        problems.append({"text": problem_text})

        if len(problems) % 10000 == 0:
            print(f"Generated {len(problems):,} / {TARGET_COUNT:,} problems")

    random.shuffle(problems)

    output_path = os.path.join(os.path.dirname(__file__), "math_word_problems.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for problem in problems:
            f.write(json.dumps(problem) + '\n')

    print(f"Saved {len(problems):,} problems to {output_path}")


if __name__ == "__main__":
    generate_problems()
