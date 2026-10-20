"""The domain-free part of running an agent: a request to the model, the teacher as the model, and
the loop that drives many sessions at once.

A session is a small state machine with one pending Request (None when it is finished) and a
receive(text) method; run_sessions batches the pending requests of all sessions into one model
call per round. The math tutor (math_tutor/harness.py) and the coding agent (code_agent/harness.py)
each bring their own sessions.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

FULL_PRECISION_MEMORY_GB = 30   # a 12B teacher in bf16 needs ~26 GB: below this, 4 bits (NF4)


@dataclass
class Request:
    kind: str                 # what the session asks for, e.g. "route", "solution", "scope", "answer"
    context: str              # the text the model reads (the domain's teacher prompt is put around it)
    prefix: str = ""          # what the model's turn starts with
    max_tokens: int = 120
    temperature: float = 0.0
    stop: List[str] = field(default_factory=list)


class TeacherModel:
    """The teacher as the agent's model: the domain's teacher prompt around each request's context.
    At most `batch` requests per model call (8 keeps a 12 GB card off its memory limit)."""

    def __init__(self, handler, domain, batch: int = 8):
        self.handler, self.domain, self.batch = handler, domain, batch

    def generate(self, requests: List[Request]) -> List[str]:
        groups: Dict[tuple, List[int]] = {}
        for index, request in enumerate(requests):
            groups.setdefault((request.max_tokens, request.temperature, tuple(request.stop)), []).append(index)
        texts = [""] * len(requests)
        chunks = [(key, indices[start:start + self.batch]) for key, indices in groups.items()
                  for start in range(0, len(indices), self.batch)]
        for (max_tokens, temperature, stop), indices in chunks:
            prompts = [self.domain.teacher_prompt(requests[i].context) for i in indices]
            outputs = self.handler.generate_texts(prompts, max_new_tokens=max_tokens, temperature=temperature,
                                                  stop_strings=list(stop) or None, prefixes=[requests[i].prefix for i in indices])
            for i, output in zip(indices, outputs):
                texts[i] = output
        return texts


def teacher_setup(conversations_path: Path) -> bool:
    """Whether the teacher of this pass runs in 4 bits (True) or full precision (False): full
    precision when the card has room for it. The first run writes the choice next to the
    conversations file (<name>.setup.json) and every later run must match it, so one file never
    mixes replies of two teacher setups (the teacher data is read back with the same setup)."""
    import torch
    setup_path = conversations_path.with_suffix(".setup.json")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    load_in_4bit = memory_gb < FULL_PRECISION_MEMORY_GB
    if setup_path.exists():
        recorded = json.loads(setup_path.read_text(encoding="utf-8"))["load_in_4bit"]
        if recorded != load_in_4bit:
            raise SystemExit(f"{conversations_path.name} was made with load_in_4bit={recorded}, this card gives "
                             f"{load_in_4bit} ({memory_gb:.0f} GB): use the same setup, or start a new conversations file.")
        return recorded
    if conversations_path.exists():
        raise SystemExit(f"{conversations_path.name} exists without {setup_path.name}: its teacher setup is unknown.")
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(json.dumps({"load_in_4bit": load_in_4bit, "gpu_memory_gb": round(memory_gb)}), encoding="utf-8")
    return load_in_4bit


def recorded_setup(conversations_path: Path) -> bool:
    """load_in_4bit of a finished pass (for reading its replies back with the same setup)."""
    return json.loads(conversations_path.with_suffix(".setup.json").read_text(encoding="utf-8"))["load_in_4bit"]


class StudentAgentModel:
    """A trained student as the agent's model. It writes its whole reply itself (it was trained on
    whole replies), so the prefix the teacher's turn was started with is taken off the front, as the
    sessions expect. A reply that does not start with it is passed on as it is: the session's check
    then fails and asks again."""

    def __init__(self, student, max_steps: int):
        self.student, self.max_steps = student, max_steps

    def generate(self, requests: List[Request]) -> List[str]:
        texts = []
        for request in requests:
            output = self.student.generate(request.context, max_new_tokens=self.max_steps,
                                           temperature=request.temperature).text.strip()
            texts.append(output[len(request.prefix):] if request.prefix and output.startswith(request.prefix) else output)
        return texts


def run_sessions(model, sessions: List, exit_listener=None, progress: bool = False) -> bool:
    """Drive every session to the end, batching each round's pending requests. With an
    exit_listener (Stop button / Ctrl+C), a stop is honoured between rounds: the round in progress
    finishes and the sessions keep everything said so far. False when stopped before the end."""
    from time import time
    round_number = 0
    while True:
        active = [session for session in sessions if not session.done]
        if not active:
            return True
        if exit_listener is not None and exit_listener.check_exit():
            return False
        round_number += 1
        started = time()
        outputs = model.generate([session.pending for session in active])
        for session, output in zip(active, outputs):
            session.receive(output)
        if progress:
            finished = sum(session.done for session in sessions)
            print(f"    round {round_number}: {len(active)} replies in {time() - started:.0f} s, "
                  f"{finished}/{len(sessions)} sessions finished", flush=True)
