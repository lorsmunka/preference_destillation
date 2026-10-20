"""Talk to a trained math-tutor student: type a word problem, then follow-ups (a new task, a
what-if, or a question about the working). The harness does what it does in the pass: the
student's first tag routes the message; a solution is checked line by line and asked for again
once; an explanation is checked and asked for again once; an off-topic message gets the
harness's fixed reply and never enters the student's context.

    python -m library.domain.math_tutor.demo                 # the slice student trained on all groups
    python -m library.domain.math_tutor.demo <run name>
"""

import sys

from library.agent.runner import Request, StudentAgentModel
from .checks import check_explanation, check_lines, complete_results
from .conversation import Conversation, parse_route
from .harness import EXPLANATION_PREFIX, OUT_OF_SCOPE_REPLY, SOLUTION_PREFIX

DEFAULT_RUN = "slice-mix-both-kd-s1"   # the copies-vs-new test's "both" student (lowest test error, 2026-10-17)


def ask(model, request: Request, check, retries: int = 1):
    """The model's reply to a request, asked again once (sampled) when the check finds a problem."""
    for attempt in range(retries + 1):
        if attempt:
            request = Request(request.kind, request.context, request.prefix, request.max_tokens, 0.7, request.stop)
        reply = request.prefix + model.generate([request])[0]
        problem = check(reply)
        if problem is None:
            return reply, None
    return reply, problem


def explanation_problem(reply: str):
    if "</explanation>" not in reply:
        return "cut off: no </explanation>"
    return check_explanation(reply[len(EXPLANATION_PREFIX):].replace("</explanation>", "").strip())


def answer(model, conversation: Conversation, message: str) -> str:
    """One agent turn for a typed message; the conversation is updated as the harness does."""
    route = parse_route(model.generate([Request("route", conversation.route_context(message), max_tokens=24)])[0])
    if route == "out_of_scope":
        return OUT_OF_SCOPE_REPLY + "   (the harness)"
    if route == "explanation":
        reply, problem = ask(model, Request("explanation", conversation.explain_context(message), EXPLANATION_PREFIX,
                                            stop=["</explanation>"]), explanation_problem)
        conversation.add_explanation(message, reply[len(EXPLANATION_PREFIX):].replace("</explanation>", "").strip())
    else:   # a solution, or unclear: the harness treats it as a task
        reply, problem = ask(model, Request("solution", conversation.solve_context(message), SOLUTION_PREFIX,
                                            stop=["</solution>"]), check_lines)
        conversation.add_solution(message, complete_results(reply).replace("<solution>", "").replace("</solution>", "").strip())
    return reply + (f"\n   (check after the second try: {problem})" if problem else "")


def main():
    from library import RunStore
    from library.domain import get_domain
    from library.model.student import StudentModel

    store = RunStore()
    run = store.get(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN)
    domain = get_domain(run.domain)
    _, checkpoint_path = store.resolve_checkpoint(run.name, "latest")
    model = StudentAgentModel(StudentModel.from_run(run, checkpoint_path, domain=domain), domain.max_steps)
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
