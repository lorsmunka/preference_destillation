"""Harness state machine with an adversarial fake model: no-tag routes, tasks routed out of scope or to an
explanation, failing explanations, what-ifs routed to an explanation. Also: is each request's context what the
Conversation rules say?"""
import sys, json, random
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.questions import generator
from library.domain.math_tutor.harness import TutorSession, TemplatePupil, run_sessions, summarise, transcript_page, OUT_OF_SCOPE_REPLY
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
test_items = [i for i in items if i["split"] == "test"][:300]
gen = generator()
rng = random.Random(1)
log = Counter()
class Adversary:
    def generate(self, requests):
        out = []
        for r in requests:
            roll = rng.random()
            if r.kind == "route":
                if not r.prefix and roll < 0.15: out.append("Sure! Let me think about that."); log["no tag"] += 1
                elif r.prefix and roll < 0.5: out.append("message>huh"); log["no tag after <"] += 1
                elif roll < 0.25: out.append("<out_of_scope>"); log["oos"] += 1
                elif roll < 0.55: out.append(("" if r.prefix else "<") + "explanation>Because"); log["exp"] += 1
                else: out.append(("" if r.prefix else "<") + "solution>\nx = 1"); log["sol"] += 1
            elif r.kind == "solution":
                out.append(rng.choice(["x = 3\ny = 4\nz = 3 + 4\nAnswer: 7\n</solution>", "a = 2\nAnswer: 2\n</solution>", "1. bad\n</solution>"]))
            else:
                out.append(rng.choice(["I missed it, sorry.</explanation>", "", "Because 2 + 2 = 4.</explanation>"]))
        return out
sessions = [TutorSession(TemplatePupil(i, gen, seed=3)) for i in test_items]
run_sessions(Adversary(), sessions)
s = summarise(sessions)
print(log); print({k: v for k, v in s.items() if k != "routes_by_kind"})
# invariants
problems = Counter()
for se in sessions:
    history = se.conversation.history(with_thread=True)
    for t in se.transcript:
        if t.get("answered_by") == "harness" and t["message"] in history:
            problems["an out-of-scope message is in the model's context"] += 1
    # every pupil task event of the item appears in the transcript
    tasks = [e for e in se.pupil.events if e["type"] == "task"]
    if sum(1 for t in se.transcript if t["event"] == "task") != len(tasks): problems["task events skipped"] += 1
print("invariant problems:", problems)
page = transcript_page({"summary": s, "model": "fake", "conversations": [{"item_id": x.pupil.item["id"], "slice": x.pupil.item["slice"], "turns": x.transcript} for x in sessions]})
print("page", len(page))
