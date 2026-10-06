"""Conversation layer: what the tutor model reads and writes, turn by turn. The same rules
build the training corpus and run the live harness, so they cannot drift apart.

Past turns carry their type; the newest input is an untyped <message>:

    <task>Ravi buys 6 tetras at $2 each. What does he pay?</task>
    <solution>
    cost = 2 * 6 = 12
    Answer: 12
    </solution>
    <question>Why did you multiply?</question>            (side thread, while it is open)
    <explanation>Each of the 6 tetras costs $2 ...</explanation>
    <message>He also buys food for $5. How much now?</message>

The model answers with one block, <solution>...</solution> or <explanation>...</explanation>;
its first token is the route. When a message turns out to be a task, the side thread leaves
the model's context (the harness keeps the full transcript): the solution is produced from
the main line only. Turn types:

    route    main line + open thread + <message>   -> the opening tag only
    solve    main line + <message>                  -> <solution> ... </solution>
    explain  main line + thread so far + <message>  -> <explanation> ... </explanation>
"""

from typing import Dict, Iterator, List, Optional, Tuple

from .annotations import NEEDS_STORY_NAMES, render

SOLUTION, EXPLANATION = "<solution>", "<explanation>"
CLOSING = {SOLUTION: "</solution>", EXPLANATION: "</explanation>"}


def block(tag: str, text: str, multiline: bool = False) -> str:
    return f"<{tag}>\n{text}\n</{tag}>" if multiline else f"<{tag}>{text}</{tag}>"


class Conversation:
    """One tutoring session: the main line of task/solution pairs and the open side thread."""

    def __init__(self):
        self.main_line: List[Tuple[str, str]] = []   # (task, solution body)
        self.thread: List[Tuple[str, str]] = []      # (question, explanation text)

    # --- what the model reads ------------------------------------------------------
    def history(self, with_thread: bool) -> str:
        parts = []
        for task, solution in self.main_line:
            parts += [block("task", task), block("solution", solution, multiline=True)]
        if with_thread:
            for question, explanation in self.thread:
                parts += [block("question", question), block("explanation", explanation)]
        return "\n".join(parts)

    def context(self, message: str, with_thread: bool) -> str:
        history = self.history(with_thread)
        return (history + "\n" if history else "") + block("message", message)

    def route_context(self, message: str) -> str:
        return self.context(message, with_thread=True)

    def solve_context(self, message: str) -> str:
        return self.context(message, with_thread=False)

    def explain_context(self, message: str) -> str:
        return self.context(message, with_thread=True)

    # --- what the model wrote -------------------------------------------------------
    def add_solution(self, task: str, solution: str) -> None:
        self.thread = []            # a new task closes the side thread
        self.main_line.append((task, solution))

    def add_explanation(self, question: str, explanation: str) -> None:
        self.thread.append((question, explanation))


def solution_target(body: str) -> str:
    return block("solution", body, multiline=True)


def explanation_target(text: str) -> str:
    return block("explanation", text)


def solve_turns(item: Dict, annotation: str) -> Iterator[Dict]:
    """One solve record per turn of an item: the context the model reads and the gold
    solution block in the given annotation format."""
    solutions = render(item, annotation)
    tasks = [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]]
    conversation = Conversation()
    for turn, (task, solution) in enumerate(zip(tasks, solutions)):
        yield {
            "text": conversation.solve_context(task),
            "target": solution_target(solution),
            "turn_type": "solve",
            "annotation": annotation,
            "answer": item["answers"][turn],
            "item_id": item["id"],
            "family": item["family"],
            "turn": turn,
            "slice": item["slice"],
            "shape": item["shape"].split("|")[turn],
            "problem_shape": item["problem_shape"],
            "split": item["split"],
        }
        conversation.add_solution(task, solution)


def usable(item: Dict, annotation: str) -> bool:
    """Named and code labels come from the writer's variable names, which rewordings
    inherit from their original story: skip rewordings for those. Facts-first labels exist
    only for turns the label pass verified."""
    if annotation == "facts":
        try:
            render(item, "facts")
            return True
        except KeyError:
            return False
    return not (annotation in NEEDS_STORY_NAMES and str(item.get("source", "")).startswith("reword:"))
