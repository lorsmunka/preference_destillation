"""Code recall corpus — a small, randomly *structured* Python program with numbered lines,
then one question whose answer is a line number, an identifier or a literal read off the
program. Ground truth is known from the generator and stored with the record.

    python -m library.domain.code_recall.corpus        # writes corpus.jsonl here

Why the programs are not templates: each one is grown from a grammar — 2-5 functions with
random bodies (assignments, arithmetic, for/while loops, if/elif branches, nested one level,
calls to earlier functions, returns), random identifiers from a pool of ~400 names, random
blank lines and comments that mention other identifiers as distractors, and a main section.
Structure, positions, names and lengths all vary, so nothing about a question's answer can
be read from its template; it has to be found in the text. Every program is checked with
ast.parse, so it is valid Python.
"""

import ast
import json
import random
from typing import Dict, List, Optional

NOUNS = ["count", "total", "index", "value", "item", "result", "score", "limit", "step", "size",
         "width", "height", "depth", "offset", "buffer", "cache", "entry", "record", "row", "col",
         "sum", "prod", "acc", "flag", "level", "rate", "delta", "weight", "bias", "seed",
         "batch", "chunk", "page", "line", "token", "word", "node", "edge", "path", "key",
         "price", "tax", "cost", "budget", "balance", "debt", "credit", "stock", "order", "cart"]
PREFIXES = ["", "", "", "raw_", "new_", "old_", "max_", "min_", "cur_", "tmp_", "base_", "next_"]
SUFFIXES = ["", "", "", "s", "_count", "_total", "_list", "_map", "_id", "_len", "_sum", "2"]
VERBS = ["load", "save", "parse", "build", "compute", "update", "check", "scan", "merge", "split",
         "filter", "count", "render", "encode", "decode", "reset", "apply", "collect", "normalize",
         "validate", "process", "score", "rank", "fetch", "store", "prepare", "finish", "summarize"]
OBJECTS = ["data", "rows", "items", "config", "state", "batch", "records", "input", "output", "totals",
           "scores", "cache", "buffer", "table", "graph", "queue", "stack", "model", "report", "index"]
COMMENT_TEMPLATES = ["# TODO: revisit {a}", "# {a} must stay positive", "# see {a} and {b}",
                     "# handles the {a} case", "# note: {b} is updated elsewhere", "# {a}: legacy name"]


class ProgramBuilder:
    """Grows one random program line by line while recording the facts questions can ask about."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.lines: List[str] = []
        self.functions: List[dict] = []      # {"name", "params", "line", "returns", "calls"}
        self.assignments: List[dict] = []    # {"name", "line", "value", "scope"}
        self.while_loops: List[dict] = []    # {"line", "variable", "direction", "scope"}
        self.for_loops: List[dict] = []      # {"line", "variable", "iterable", "scope"}
        self.ifs: List[dict] = []            # {"line", "left", "right", "op", "scope"}
        self.used_names = set()

    # ── naming ────────────────────────────────────────────────────────
    def variable_name(self) -> str:
        for _ in range(50):
            name = self.rng.choice(PREFIXES) + self.rng.choice(NOUNS) + self.rng.choice(SUFFIXES)
            if name not in self.used_names and not name[0].isdigit():
                self.used_names.add(name)
                return name
        return f"var_{len(self.used_names)}"

    def function_name(self) -> str:
        for _ in range(50):
            name = f"{self.rng.choice(VERBS)}_{self.rng.choice(OBJECTS)}"
            if name not in self.used_names:
                self.used_names.add(name)
                return name
        return f"func_{len(self.used_names)}"

    # ── emitting ──────────────────────────────────────────────────────
    def emit(self, text: str, indent: int) -> int:
        self.lines.append("    " * indent + text)
        return len(self.lines)          # 1-based line number

    def maybe_blank_or_comment(self, indent: int, names: List[str]):
        roll = self.rng.random()
        if roll < 0.12:
            self.lines.append("")
        elif roll < 0.24 and names:
            a, b = self.rng.choice(names), self.rng.choice(names)
            self.emit(self.rng.choice(COMMENT_TEMPLATES).format(a=a, b=b), indent)

    def literal(self) -> str:
        return str(self.rng.choice([0, 1, 2, 3, 5, 7, 8, 10, 12, 16, 20, 25, 32, 42, 50, 64, 99, 100]))

    def block(self, indent: int, scope: str, available: List[str], depth: int = 0) -> List[str]:
        """A random body. Returns the names it assigned (available afterwards)."""
        names = list(available)
        for _ in range(self.rng.randint(2, 3) if depth == 0 else self.rng.randint(1, 2)):
            self.maybe_blank_or_comment(indent, names)
            if depth == 0:
                kinds = ["assign", "assign", "arith", "for", "while", "if", "call"]
            elif depth == 1:
                kinds = ["assign", "arith", "arith", "if"]
            else:
                kinds = ["assign", "arith", "arith"]       # no nesting past two levels
            kind = self.rng.choice(kinds)
            if kind == "assign" or (kind in ("arith", "call", "if", "while") and not names):
                name = self.variable_name()
                value = self.literal()
                line = self.emit(f"{name} = {value}", indent)
                self.assignments.append({"name": name, "line": line, "value": value, "scope": scope})
                names.append(name)
            elif kind == "arith":
                target = self.rng.choice(names)
                other = self.rng.choice([n for n in names if n != target] + [self.literal()])
                op = self.rng.choice(["+", "-", "*", "//", "%"])
                line = self.emit(f"{target} = {target} {op} {other}", indent)
                self.assignments.append({"name": target, "line": line, "value": None, "scope": scope})
            elif kind == "for":
                variable = self.variable_name()
                iterable = self.rng.choice([f"range({self.rng.choice(names + [self.literal()])})", f"range({self.literal()})",
                                            self.rng.choice(names) + "s" if names else "range(3)"])
                line = self.emit(f"for {variable} in {iterable}:", indent)
                self.for_loops.append({"line": line, "variable": variable, "iterable": iterable, "scope": scope})
                names.append(variable)
                self.block(indent + 1, scope, names, depth + 1)
            elif kind == "while":
                variable = self.rng.choice(names)
                direction = self.rng.choice(["decrement", "increment"])
                bound = self.literal()
                line = self.emit(f"while {variable} {'>' if direction == 'decrement' else '<'} {bound}:", indent)
                self.while_loops.append({"line": line, "variable": variable, "direction": direction, "scope": scope})
                inner = self.block(indent + 1, scope, names, depth + 1)
                self.emit(f"{variable} {'-=' if direction == 'decrement' else '+='} {self.rng.choice(['1', '1', '2', '3'])}", indent + 1)
            elif kind == "if":
                left = self.rng.choice(names)
                right = self.rng.choice([n for n in names if n != left] + [self.literal()])
                op = self.rng.choice(["==", "!=", "<", ">", "<=", ">="])
                line = self.emit(f"if {left} {op} {right}:", indent)
                self.ifs.append({"line": line, "left": left, "right": right, "op": op, "scope": scope})
                self.block(indent + 1, scope, names, depth + 1)
                if self.rng.random() < 0.4:
                    self.emit("else:", indent)
                    self.block(indent + 1, scope, names, depth + 1)
            elif kind == "call" and [f for f in self.functions if f["name"] != scope]:
                callee = self.rng.choice([f for f in self.functions if f["name"] != scope])   # no self-calls
                args = ", ".join(self.rng.choice(names + [self.literal()]) for _ in callee["params"])
                target = self.variable_name()
                line = self.emit(f"{target} = {callee['name']}({args})", indent)
                self.assignments.append({"name": target, "line": line, "value": None, "scope": scope})
                callee["calls"].append({"line": line, "scope": scope})
                names.append(target)
        return names

    def function(self):
        name = self.function_name()
        params = [self.variable_name() for _ in range(self.rng.randint(0, 3))]
        self.used_names.update(params)
        line = self.emit(f"def {name}({', '.join(params)}):", 0)
        record = {"name": name, "params": params, "line": line, "returns": None, "calls": []}
        self.functions.append(record)
        names = self.block(1, name, params)
        returned = self.rng.choice(names) if names else self.literal()
        self.emit(f"return {returned}", 1)
        record["returns"] = returned
        self.lines.append("")

    def build(self):
        for _ in range(self.rng.randint(2, 4)):
            self.function()
        self.emit("# main", 0)
        self.block(0, "main", [])
        while self.lines and self.lines[-1] == "":
            self.lines.pop()
        return self


QUESTION_TYPES = ["while_line", "for_line", "if_line", "initial_value", "returns", "caller_line",
                  "first_param", "param_count", "function_count", "last_assignment", "def_line"]


def make_question(rng: random.Random, program: ProgramBuilder):
    """One question with a unique answer derivable from the text. Returns (type, question, answer)."""
    candidates = []
    seen_while = {}
    for loop in program.while_loops:
        seen_while.setdefault((loop["variable"], loop["direction"]), []).append(loop)
    for (variable, direction), loops in seen_while.items():
        if len(loops) == 1:
            candidates.append(("while_line", f"On which line is the while loop that {direction}s `{variable}`?", str(loops[0]["line"])))
    for loop in program.for_loops:
        if sum(1 for other in program.for_loops if other["variable"] == loop["variable"]) == 1:
            candidates.append(("for_line", f"On which line does the for loop over `{loop['variable']}` start?", str(loop["line"])))
    for branch in program.ifs:
        same = [b for b in program.ifs if b["left"] == branch["left"] and b["op"] == branch["op"] and b["right"] == branch["right"]]
        if len(same) == 1:
            candidates.append(("if_line", f"On which line is the condition `{branch['left']} {branch['op']} {branch['right']}` tested?", str(branch["line"])))
    firsts = {}
    for assignment in program.assignments:
        if assignment["value"] is not None:
            firsts.setdefault(assignment["name"], assignment)
    for name, assignment in firsts.items():
        candidates.append(("initial_value", f"What value is `{name}` first set to?", assignment["value"]))
    for function in program.functions:
        candidates.append(("returns", f"What does `{function['name']}` return?", function["returns"]))
        candidates.append(("def_line", f"On which line is `{function['name']}` defined?", str(function["line"])))
        candidates.append(("param_count", f"How many parameters does `{function['name']}` take?", str(len(function["params"]))))
        if function["params"]:
            candidates.append(("first_param", f"What is the first parameter of `{function['name']}`?", function["params"][0]))
        if len(function["calls"]) == 1:
            candidates.append(("caller_line", f"On which line is `{function['name']}` called?", str(function["calls"][0]["line"])))
    candidates.append(("function_count", "How many functions are defined?", str(len(program.functions))))
    lasts = {}
    for assignment in program.assignments:
        lasts[assignment["name"]] = assignment["line"]
    for name, line in lasts.items():
        if sum(1 for a in program.assignments if a["name"] == name) > 1:
            candidates.append(("last_assignment", f"On which line is `{name}` assigned for the last time?", str(line)))
    kind = rng.choice(sorted({c[0] for c in candidates}))          # balance across question types
    return rng.choice([c for c in candidates if c[0] == kind])


def make_record(rng: random.Random) -> Dict:
    for _ in range(20):
        program = ProgramBuilder(rng).build()
        source = "\n".join(program.lines)
        try:
            ast.parse(source)
        except SyntaxError:
            continue
        if not 15 <= len(program.lines) <= 72:
            continue
        kind, question, answer = make_question(rng, program)
        numbered = "\n".join(f"{index + 1:>2}: {line}" for index, line in enumerate(program.lines))
        return {"text": f"{numbered}\n\nQuestion: {question}", "answer": answer, "attribute": kind,
                "line_count": len(program.lines)}
    raise RuntimeError("could not build a valid program")


ALL_LINE_NUMBERS = [str(n) for n in range(1, 81)]


def all_possible_answers() -> List[str]:
    """Every string the answer can be: line numbers, literals, identifiers the pool can form."""
    answers = set(ALL_LINE_NUMBERS) | {str(v) for v in [0, 1, 2, 3, 5, 7, 8, 10, 12, 16, 20, 25, 32, 42, 50, 64, 99, 100]}
    for prefix in PREFIXES:
        for noun in NOUNS:
            for suffix in SUFFIXES:
                answers.add(prefix + noun + suffix)
    for verb in VERBS:
        for obj in OBJECTS:
            answers.add(f"{verb}_{obj}")
    return sorted(answers)


def build_corpus(path: str, count: Optional[int] = None, seed: int = 0) -> None:
    count = count or 10000
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as file:
        for _ in range(count):
            file.write(json.dumps(make_record(rng), ensure_ascii=False) + "\n")
    print(f"Wrote {count} code recall records -> {path}")


if __name__ == "__main__":
    import os
    build_corpus(os.path.join(os.path.dirname(__file__), "corpus.jsonl"))
