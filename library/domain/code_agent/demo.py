"""Talk to a trained coding-agent student: a project version is in its context, you type a
request ("Where is the code that cancels an order?", "Explain how shipping is priced."), and the
harness runs both steps with the student: its scope, the narrowed lines, its answer (explanation
or found), each checked and asked again once as in the pass.

    python -m library.domain.code_agent.demo                        # the newest coding-agent student, shop version 0
    python -m library.domain.code_agent.demo <run name> <project> <version>
"""

import sys

from library.agent.runner import StudentAgentModel, run_sessions
from .harness import CodeAgentSession
from .project import load_base, make_version, render


def newest_run():
    from library import RunStore
    runs = RunStore().filter(prefix="code-agent-", status="completed")
    if not runs:
        raise SystemExit("No trained coding-agent student yet: python experiments/code_agent_trial.py")
    return max(runs, key=lambda run: (run.run_dir / "info.json").stat().st_mtime)


def load_student(run):
    from library import RunStore
    from library.domain import get_domain
    from library.model.student import StudentModel
    _, checkpoint_path = RunStore().resolve_checkpoint(run.name, "latest")
    domain = get_domain(run.domain)   # the variant the student was trained on
    return StudentModel.from_run(run, checkpoint_path, domain=domain), domain


def show_turn(version, turn):
    print(f"\n[scope] {turn.get('scope', '').strip()}" + (f"   (check: {turn['scope_check']})" if turn.get("scope_check") else ""))
    if turn.get("ranges"):
        print(render(version.files, {name: [tuple(span) for span in spans] for name, spans in turn["ranges"].items()}))
    if turn.get("output"):
        print(f"\n[answer] {turn['output']}" + (f"   (check: {turn['check']})" if turn.get("check") else ""))
    for rejected in turn.get("rejected", []):
        print(f"  (asked again after: {rejected['problem']})")


def main():
    from library import RunStore
    run = RunStore().get(sys.argv[1]) if len(sys.argv) > 1 else newest_run()
    project, number = (sys.argv[2], int(sys.argv[3])) if len(sys.argv) > 3 else ("shop", 0)
    student, domain = load_student(run)
    version = make_version(load_base(project), number)
    model = StudentAgentModel(student, domain.max_steps)
    print(f"student {run.name}; project {version.id}: {', '.join(version.files)}")
    print("Type a request (find or explain something), or q to quit.")
    while True:
        message = input("\n> ").strip()
        if message.lower() in ("q", "quit", ""):
            break
        kind = "explain" if message.lower().startswith(("explain", "how", "what", "why", "tell", "help", "can you explain")) else "find"
        item = {"id": "demo", "project": project, "version": number, "split": "demo", "kind": kind, "source": "typed",
                "message": message, "truth": []}
        session = CodeAgentSession(item, version)
        run_sessions(model, [session])
        show_turn(version, session.turn)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
