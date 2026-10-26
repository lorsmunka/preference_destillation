"""Talk to a trained math-tutor student: type a word problem, then follow-ups (a new task, a
what-if, or a question about the working). The harness does what it does in the pass: the
student's first tag routes the message; a solution is checked line by line and asked for again
once; an explanation is checked and asked for again once; an off-topic message gets the
harness's fixed reply and never enters the student's context.

    python -m library.domain.math_tutor.demo                 # the best student of the full run (cloud, 2026-10-26)
    python -m library.domain.math_tutor.demo <run name>
    python tutor_chat.py                                     # the same in a local chat page, step by step
"""

import sys
from typing import Dict, List

from library.agent.runner import Request, StudentAgentModel
from .checks import check_explanation, check_lines, complete_results
from .conversation import Conversation, parse_route
from .harness import EXPLANATION_PREFIX, OUT_OF_SCOPE_REPLY, SOLUTION_PREFIX

DEFAULT_RUN = "math-full-kd-l-s1"   # kd at 53M: 58% of test answers right (findings F-M1)


def ask(model, request: Request, check, retries: int = 1):
    """The model's reply to a request, asked again once (sampled) when the check finds a problem,
    with every attempt as a call record (what it read, what it wrote, what the check found)."""
    calls = []
    for attempt in range(retries + 1):
        if attempt:
            request = Request(request.kind, request.context, request.prefix, request.max_tokens, 0.7, request.stop)
        reply = request.prefix + model.generate([request])[0]
        problem = check(reply)
        calls.append(call_record(request, reply, problem))
        if problem is None:
            break
    return reply, problem, calls


def call_record(request: Request, reply: str, problem=None) -> Dict:
    return {"kind": request.kind, "context": request.context, "temperature": request.temperature, "reply": reply,
            "problem": problem}


def explanation_problem(reply: str):
    if "</explanation>" not in reply:
        return "cut off: no </explanation>"
    return check_explanation(reply[len(EXPLANATION_PREFIX):].replace("</explanation>", "").strip())


def take_turn(model, conversation: Conversation, message: str) -> Dict:
    """One agent turn for a typed message, step by step: the route, every model call and the reply.
    The conversation is updated as the harness does."""
    route_request = Request("route", conversation.route_context(message), max_tokens=24)
    route_text = model.generate([route_request])[0]
    route = parse_route(route_text)
    calls: List[Dict] = [call_record(route_request, route_text)]
    if route == "out_of_scope":
        return {"route": route, "calls": calls, "reply": OUT_OF_SCOPE_REPLY, "problem": None}
    if route == "explanation":
        reply, problem, attempts = ask(model, Request("explanation", conversation.explain_context(message), EXPLANATION_PREFIX,
                                                      stop=["</explanation>"]), explanation_problem)
        conversation.add_explanation(message, reply[len(EXPLANATION_PREFIX):].replace("</explanation>", "").strip())
    else:   # a solution, or unclear: the harness treats it as a task
        reply, problem, attempts = ask(model, Request("solution", conversation.solve_context(message), SOLUTION_PREFIX,
                                                      stop=["</solution>"]), check_lines)
        conversation.add_solution(message, complete_results(reply).replace("<solution>", "").replace("</solution>", "").strip())
    return {"route": route, "calls": calls + attempts, "reply": reply, "problem": problem}


def answer(model, conversation: Conversation, message: str) -> str:
    """One agent turn for a typed message, as the text the pupil sees."""
    turn = take_turn(model, conversation, message)
    if turn["route"] == "out_of_scope":
        return turn["reply"] + "   (the harness)"
    return turn["reply"] + (f"\n   (check after the second try: {turn['problem']})" if turn["problem"] else "")


def load_student(run_name: str = DEFAULT_RUN):
    """(the run, the student as the agent's model) for a trained run."""
    from library import RunStore
    from library.domain import get_domain
    from library.model.student import StudentModel

    store = RunStore()
    run = store.get(run_name)
    domain = get_domain(run.domain)
    _, checkpoint_path = store.resolve_checkpoint(run.name, "latest")
    return run, StudentAgentModel(StudentModel.from_run(run, checkpoint_path, domain=domain), domain.max_steps)


def main():
    run, model = load_student(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN)
    conversation = Conversation()
    print(f"student {run.name}. Type a word problem, then follow-ups; 'new' starts over, 'q' quits.")
    while True:
        message = input("\npupil> ").strip()
        if message.lower() in ("q", "quit", ""):
            break
        if message.lower() == "new":
            conversation = Conversation()
            continue
        print(answer(model, conversation, message))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
