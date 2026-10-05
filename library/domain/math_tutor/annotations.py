"""Annotation layer: how a turn's solution is written down. Every format is rendered from
the item's executed solve() trace, so the gold version of each format is correct by
construction. A new format is one more function in RENDERERS; the conversation layer only
asks for a renderer by name, so formats can be swapped without touching the items.

  answer   Answer: 5
  working  4 * 2 = 8            (each computed step, values filled in)
           13 - 8 = 5
           Answer: 5
  named    cost of more tetras = 4 * 2 = 8
           money left = 13 - 8 = 5
           Answer: 5
  code     cost_of_more_tetras = 4 * price     (the student writes code, the harness runs it;
           money_left = left - cost_of_more_tetras   earlier turns' names stay usable)
           Answer: money_left

The labels in "named" and "code" are the writer's variable names. Rewordings reuse their
original's code, so their names describe the original story: render those two formats only
for items whose source is not "reword:" until a relabel pass has renamed them.
"""

import ast
from typing import Callable, Dict, List

from .functions import Execution, execute
from .items import REMAINDER_SLICE


def turn_executions(item: Dict) -> List[Execution]:
    """Run every turn of an item, each follow-up on the previous turn's state."""
    allow_remainders = item["slice"] == REMAINDER_SLICE
    sources = [item["solve"]] + [follow_up["solve"] for follow_up in item["follow_ups"]]
    executions, state = [], None
    for source in sources:
        execution = execute(source, state, allow_remainders, trace=True)
        executions.append(execution)
        state = execution.result
    return executions


def answer_name(source: str) -> str:
    """Which name holds the answer in a solve(): the variable returned as 'answer', or
    'answer' itself when the return computes it."""
    returned = ast.parse(source).body[0].body[-1].value
    for key, value in zip(returned.keys, returned.values):
        if isinstance(key, ast.Constant) and key.value == "answer":
            if isinstance(value, ast.Name):
                return value.id
            if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant):
                return value.slice.value
    return "answer"


def label(name: str) -> str:
    return "result" if name == "answer" else name.replace("_", " ")


def shown(expression: str, item: Dict) -> str:
    """Division is written / outside the remainder slice (it is always exact there)."""
    return expression if item["slice"] == REMAINDER_SLICE else expression.replace("//", "/")


def render_answer(item: Dict, turn: int, executions: List[Execution]) -> str:
    return f"Answer: {executions[turn].result['answer']}"


def render_working(item: Dict, turn: int, executions: List[Execution]) -> str:
    lines = [f"{shown(step.filled, item)} = {step.value}" for step in executions[turn].steps if step.computes]
    return "\n".join(lines + [render_answer(item, turn, executions)])


def render_named(item: Dict, turn: int, executions: List[Execution]) -> str:
    lines = [f"{label(step.name)} = {shown(step.filled, item)} = {step.value}"
             for step in executions[turn].steps if step.computes]
    return "\n".join(lines + [render_answer(item, turn, executions)])


def render_code(item: Dict, turn: int, executions: List[Execution]) -> str:
    source = item["solve"] if turn == 0 else item["follow_ups"][turn - 1]["solve"]
    lines = [f"{step.name} = {step.code}" for step in executions[turn].steps]
    return "\n".join(lines + [f"Answer: {answer_name(source)}"])


RENDERERS: Dict[str, Callable[[Dict, int, List[Execution]], str]] = {
    "answer": render_answer,
    "working": render_working,
    "named": render_named,
    "code": render_code,
}
NEEDS_STORY_NAMES = {"named", "code"}   # wrong for rewordings until they are relabelled


def render(item: Dict, annotation: str) -> List[str]:
    """The gold solution text of every turn of an item, in one annotation format."""
    if annotation in NEEDS_STORY_NAMES and str(item.get("source", "")).startswith("reword:"):
        raise ValueError(f"{item['id']}: '{annotation}' labels come from the original story; relabel rewordings first")
    executions = turn_executions(item)
    renderer = RENDERERS[annotation]
    return [renderer(item, turn, executions) for turn in range(len(executions))]
