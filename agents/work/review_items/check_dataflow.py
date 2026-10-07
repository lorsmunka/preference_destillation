import ast, json, sys, collections
sys.path.insert(0, r"D:\preference_destillation")

ITEMS = r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl"
items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def reads_state_in(node):
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "state" for n in ast.walk(node))


def state_keys_in(node):
    return [n.slice.value for n in ast.walk(node) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "state"]


counts = collections.Counter()
examples = collections.defaultdict(list)
for item in items:
    for index, follow_up in enumerate(item["follow_ups"], start=1):
        function = ast.parse(follow_up["solve"]).body[0]
        tainted = set()
        assigned = set()
        shadow = False
        for statement in function.body[:-1]:
            if isinstance(statement, ast.Assign):
                target, value = statement.targets[0].id, statement.value
            else:
                target, value = statement.target.id, ast.BinOp(left=ast.Name(id=statement.target.id), op=statement.op, right=statement.value)
            if any(key in assigned for key in state_keys_in(value)):
                shadow = True
            if reads_state_in(value) or names_in(value) & tainted:
                tainted.add(target)
            else:
                tainted.discard(target)
            assigned.add(target)
        returned = function.body[-1].value
        keys = [k.value if k is not None else None for k in returned.keys]
        answer_node = None
        for key, value in zip(returned.keys, returned.values):
            if key is not None and key.value == "answer":
                answer_node = value
            if key is not None and any(k in assigned for k in state_keys_in(value)):
                shadow = True
        if shadow:
            counts["state key read after a same-named local was set (as_code ambiguous)"] += 1
            if len(examples["shadow"]) < 3:
                examples["shadow"].append((item["id"], follow_up["solve"]))
        if "answer" not in keys:
            counts["follow-up never sets 'answer' (inherits previous answer)"] += 1
            if len(examples["noanswer"]) < 3:
                examples["noanswer"].append((item["id"], follow_up["question"], follow_up["solve"], item["answers"]))
            continue
        if None in keys and keys.index(None) > keys.index("answer"):
            counts["**state after 'answer' (state's answer wins)"] += 1
        if not (reads_state_in(answer_node) or names_in(answer_node) & tainted):
            counts["answer does not depend on state"] += 1
            if len(examples["indep"]) < 4:
                examples["indep"].append((item["id"], item["problem"][:150], follow_up["question"], follow_up["solve"]))
        if item["answers"][index] == item["answers"][index - 1]:
            counts["follow-up answer equals previous turn's answer"] += 1
            if len(examples["same"]) < 3:
                examples["same"].append((item["id"], follow_up["question"], follow_up["solve"]))
    # turn 0: shadowing irrelevant; check for answer being a bare literal-copied variable without op
print(sum(len(i["follow_ups"]) for i in items), "follow-up turns")
for key, value in counts.items():
    print(f"{value:6}  {key}")
for key, exs in examples.items():
    print("==", key)
    for e in exs:
        print("   ", e)
