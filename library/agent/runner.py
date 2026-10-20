"""The domain-free part of running an agent: a request to the model, the teacher as the model, and
the loop that drives many sessions at once.

A session is a small state machine with one pending Request (None when it is finished) and a
receive(text) method; run_sessions batches the pending requests of all sessions into one model
call per round. The math tutor (math_tutor/harness.py) and the coding agent (code_agent/harness.py)
each bring their own sessions.
"""

from dataclasses import dataclass, field
from typing import Dict, List


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
