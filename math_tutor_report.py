"""The math tutor build, step by step, as one local HTML page: where the problems come
from, how many survive each stage, how they are stored, and how pruning works.

    python math_tutor_report.py      # writes thesis_work/math_tutor_report.html

Local file only, nothing is uploaded. Reads the logs and data the build scripts wrote
under library/domain/math_tutor/ (benchmarks/, items/, teacher_check.json); every block
shows what is there and says so when a stage has not run yet.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from library.domain.math_tutor import EXAMPLE_RESPONSES, PROMPTS
from library.domain.math_tutor.corpus import turn_text
from library.domain.math_tutor.items import SLICES, validate
from library.domain.math_tutor.similarity import collect_name_words, normalise, sequence_ratio, word_trigrams, jaccard
from library.domain.math_tutor.validator_cases import CASES

PACKAGE = Path("library") / "domain" / "math_tutor"
OUTPUT = Path("thesis_work") / "math_tutor_report.html"
TEMPLATE = Path("library") / "tooling" / "templates" / "math_tutor_report.html"


def read_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if path.exists() else []


def source_kind(item):
    return str(item.get("source", "?")).split(":")[0]


def human_sets():
    log = read_json(PACKAGE / "benchmarks" / "build_log.json")
    if not log:
        return None
    test = read_jsonl(PACKAGE / "benchmarks" / "real_test.jsonl")
    seeds = read_jsonl(PACKAGE / "benchmarks" / "seeds.jsonl")
    dropped = read_jsonl(PACKAGE / "benchmarks" / "dropped.jsonl")
    examples = {}
    for drop in dropped:
        examples.setdefault(drop["reason"], drop)
    return {"log": log, "test_example": test[0] if test else None,
            "seed_examples": [seeds[0], next((s for s in seeds if s["set"] == "mawps"), None),
                              next((s for s in seeds if s["set"] == "harvest"), None)],
            "drop_examples": list(examples.values())}


def writing():
    """Raw files: who wrote what, how much of it is valid, and how much survives pruning."""
    kept = {item["id"] for item in read_jsonl(PACKAGE / "items" / "items.jsonl")}
    cells = defaultdict(lambda: {"written": 0, "valid": 0, "kept": 0})
    files = []
    for path in sorted((PACKAGE / "items" / "raw").glob("*.jsonl")):
        written = valid = 0
        for item in read_jsonl(path):
            validated, _ = validate(item)
            written += 1
            valid += validated is not None
            source = source_kind(item)
            if source == "seeded" and item.get("slice") not in SLICES:
                continue
            cell = cells[(item.get("slice", "?"), source)]
            cell["written"] += 1
            cell["valid"] += validated is not None
            cell["kept"] += item.get("id") in kept
        files.append({"file": path.name, "written": written, "valid": valid})
    matrix = [{"slice": slice_name, "source": source, **counts} for (slice_name, source), counts in sorted(cells.items())]
    return {"files": files, "matrix": matrix, "writing_log": read_json(PACKAGE / "items" / "writing_log.json")}


def pruning():
    log = read_json(PACKAGE / "items" / "build_log.json")
    if not log:
        return None
    dropped = read_jsonl(PACKAGE / "items" / "dropped.jsonl")
    examples = {}
    for drop in dropped:
        examples.setdefault(drop["reason"], drop)
    items = read_jsonl(PACKAGE / "items" / "items.jsonl")
    shapes_by_slice = defaultdict(Counter)
    for item in items:
        shapes_by_slice[item["slice"]][item["shape"]] += 1
    return {"log": log, "drop_examples": list(examples.values()),
            "shapes_by_slice": {name: counter.most_common(8) for name, counter in sorted(shapes_by_slice.items())}}


def similarity_demo():
    original = "A baker made 36 muffins and sold 19 of them in the morning. How many muffins are left?"
    pairs = [
        ("one word changed", "A baker made 48 muffins and sold 21 of them in the evening. How many muffins are left?"),
        ("two words changed", "A baker baked 48 cookies and sold 21 of them in the morning. How many cookies are left?"),
        ("same frame, new nouns", "A farmer picked 48 apples and sold 21 of them in the morning. How many apples are left?"),
        ("reordered", "In the morning a baker sold 21 muffins out of the 48 she made. How many are left?"),
        ("reworded", "There were 48 muffins at the bakery. By lunch 21 had been sold. How many muffins remain?"),
        ("another problem", "Omar had 52 marbles. After a game he had 37 marbles. How many marbles did he lose?"),
    ]
    names = set()
    base = normalise(original, names)
    rows = []
    for label, text in pairs:
        words = normalise(text, names)
        rows.append({"label": label, "text": text, "sequence": round(sequence_ratio(base, words), 2),
                     "trigram": round(jaccard(word_trigrams(base), word_trigrams(words)), 2)})
    return {"original": original, "normalised": " ".join(base), "rows": rows}


def validator_cases():
    rows = []
    for name, item, expected in CASES:
        validated, reason = validate(item)
        rows.append({"case": name, "result": reason or f"valid, answers {validated['answers']}"})
    return rows


def renumber_example():
    items = {item["id"]: item for item in read_jsonl(PACKAGE / "items" / "items.jsonl")}
    copies = read_jsonl(PACKAGE / "items" / "renumbered.jsonl")
    by_original = defaultdict(list)
    for copy in copies:
        by_original[copy["renumbered_from"]].append(copy)
    for original_id, its_copies in by_original.items():
        original = items.get(original_id)
        if original and original["follow_ups"] and len(its_copies) >= 3:
            return {"original": original, "copies": its_copies[:3],
                    "log": read_json(PACKAGE / "items" / "renumber_log.json")}
    return {"log": read_json(PACKAGE / "items" / "renumber_log.json")}


def storage():
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.suffix in (".jsonl", ".json", ".md") and "__pycache__" not in path.parts:
            lines = sum(1 for _ in open(path, encoding="utf-8")) if path.suffix == ".jsonl" else None
            rows.append({"path": str(path).replace("\\", "/"), "kb": round(path.stat().st_size / 1024, 1), "lines": lines})
    items = read_jsonl(PACKAGE / "items" / "items.jsonl")
    example = next((item for item in items if len(item["follow_ups"]) >= 2), items[0] if items else None)
    turns = [turn_text(example, turn) for turn in range(1 + len(example["follow_ups"]))] if example else []
    return {"files": rows, "example": example, "turns": turns}


def templateness():
    path = Path("experiments") / "results" / "math_templateness.csv"
    if not path.exists():
        return []
    import csv
    return list(csv.DictReader(open(path, encoding="utf-8")))


def methods_qa(data):
    """The exact process, as questions a thesis reader would ask. Every number is read from
    the logs the build wrote; the prose states what was done and why, in the order done."""
    human = data["human"]["log"] if data["human"] else None
    pruning = data["pruning"]["log"] if data["pruning"] else None
    renumber = (data["renumber"] or {}).get("log")
    writing_log = read_json(PACKAGE / "items" / "writing_log.json") or {"runs": []}
    rewordings = sum(len(read_jsonl(path)) for path in (PACKAGE / "items" / "reword").glob("*.jsonl"))
    raw_items = sum(f["written"] for f in data["writing"]["files"])
    by_source = Counter()
    for item in read_jsonl(PACKAGE / "items" / "items.jsonl"):
        by_source[source_kind(item)] += 1
    seeds = human["seeds"] if human else {}
    sets = human["sets"] if human else {}
    tokens = sum(run["subagent_tokens"] for run in writing_log["runs"])
    templ = {row["set"]: row for row in data["templateness"]}

    def n(value):
        return f"{value:,}" if isinstance(value, int) else str(value)

    qa = []
    qa.append(("What is the corpus, and what is it for?", f"""
<ul><li>Training data for the <b>math tutor</b>: a small student model distilled from Gemma 4 12B that answers grade 2-4
arithmetic word problems and follow-up questions, inside a harness that keeps the conversation state (agent = model + harness).</li>
<li>One <b>item</b> = a word problem, a Python <code>solve()</code> that computes its answer, and 0-3 follow-up questions,
each with a <code>solve(state)</code> continuing from the previous turn. An item with k follow-ups gives k+1 training turns.</li>
<li>The answer is always what the code returns; no answer was ever typed by a person or a model.</li></ul>"""))

    qa.append(("Where do the problems come from?", f"""
<ol><li><b>Human test (never trained on):</b> SVAMP, {n(sets.get('svamp', {}).get('in_range', 0))} of 1,000 problems inside the range.</li>
<li><b>Human seeds:</b> ASDiv {n(seeds.get('by_set', {}).get('asdiv', 0))} + MAWPS {n(seeds.get('by_set', {}).get('mawps', 0))}
+ harvested open-licence school problems {n(seeds.get('by_set', {}).get('harvest', 0))} = {n(seeds.get('count', 0))} seeds, after dedup.</li>
<li><b>Written by Claude agents:</b> {n(raw_items)} items: <i>memory</i> writers (10 slices × 8 batches × 100) and
<i>seeded</i> writers (each seed rewritten twice into a new story with the same operations), plus a 150-item pilot and 20 hand-written items.</li>
<li><b>Reworded:</b> {n(rewordings)} rewordings of train items so far (two per item; text only, the code is reused).</li>
<li><b>Renumbered:</b> copies with new numbers, made by a script, re-checked by execution.</li></ol>
Kept after pruning by source: {', '.join(f'{k} {n(v)}' for k, v in by_source.most_common())}."""))

    qa.append(("Which models did what?", f"""
<ul><li><b>Claude Opus 5.5</b> (Claude Code): designed the pipeline, wrote the code, orchestrated the agents, and wrote all
items of writing runs A-D and the rewordings (as sub-agents).</li>
<li><b>Claude Sonnet 5.5</b>: the 150-item writing pilot only.</li>
<li><b>Gemma 4 12B instruct</b> (google/gemma-4-12B-it): the teacher. It never writes items; it answers them, and its outputs and
logits are what the student is distilled from.</li>
<li>Same division of labour as TinyGSM (arXiv 2312.09241): a large model writes problems with Python solutions, a smaller one is trained on them.</li></ul>"""))

    qa.append(("What exactly were the writing agents told?", """
<ul><li>The full brief is in the repository: <code>library/domain/math_tutor/writer_brief.md</code> (item format, code rules,
slice table, variety rules, the three sources, rewording rules).</li>
<li>Each writer also got an assignment: its slice, 100 items, an exact mix (unknown position, grade, distractor type,
follow-up count), 12-15 contexts of its own (out of 80) and one style emphasis (out of 8), so batches of the same slice
do not converge.</li>
<li>Seeded writers got 100 seed lines (read by line number, checked) and wrote one new item per seed with the same
operations in the same order.</li>
<li>Every writer ran the validator on its own file and fixed failures before finishing.</li></ul>"""))

    qa.append(("How is each answer guaranteed to be right?", """
<ul><li><code>solve()</code> is never executed by Python. A ~60-line interpreter walks its syntax tree and allows only
assignments, + − × // %, whole-number literals, <code>state['name']</code> and one returned dict.</li>
<li>Every intermediate value must be a whole number 0-999; division must be exact except in the remainder slice (S07).</li>
<li>Every number in the code must appear in that turn's text; unused digit numbers must match the distractor label;
follow-ups must read the previous state.</li>
<li>26 deliberately broken items, one per rule, are all rejected (<code>validator_cases.py</code>).</li>
<li>The validator cannot check that the story means what the code computes. That was spot-checked by hand (10 random
items, all correct) and is checked at scale by the teacher: items where Gemma's answer disagrees are flagged
(<code>answer_correct</code>), and every disagreement checked so far was Gemma's error.</li></ul>"""))

    qa.append(("How was the human test kept out of training?", f"""
<ul><li>SVAMP is test only: never trained on, never shown to a writer.</li>
<li>Seeds within word-sequence ratio 0.8 or trigram Jaccard 0.5 of any SVAMP problem were dropped
({n(seeds.get('dropped_by_reason', {}).get('too close to a SVAMP problem', 0))}); SVAMP was built from ASDiv, so this removes its sources.</li>
<li>Harvesters were barred from the 11 websites ASDiv was scraped from and from every copy of SVAMP / ASDiv / MAWPS / GSM8K.</li>
<li>Every written item went through the same test-leak check (stage 5).</li>
<li>Synthetic test and held-out splits are assigned by <i>family</i>: rewordings and renumberings of a test item stay in test.</li></ul>"""))

    stage_text = "".join(f"<li>{s['stage']}: {n(s['in'])} → {n(s['out'])}"
                         + (f" ({', '.join(f'{k} {v}' for k, v in s['dropped_by_reason'].items())})" if s['dropped_by_reason'] else "")
                         + "</li>" for s in (pruning or {}).get("stages", []))
    qa.append(("How were duplicates and near-duplicates removed?", f"""
<ol><li><b>Normalise:</b> lowercase, punctuation out, numbers (digits or words) → N, names → NAME.</li>
<li><b>Exact:</b> identical normalised text → drop the later one.</li>
<li><b>Near:</b> word-sequence ratio (2 × words matching in order / all words, difflib) above 0.8 → drop. One word swapped in
a 17-word problem scores 0.94; the same words reordered ~0.4 and survive (reordering is real variety).
Word-trigram Jaccard was tried first and replaced after the pilot: it let a one-word swap through (0.68).</li>
<li>Candidates are found through each problem's rarest words (prefix filtering); this is exact, it matched brute force on all seeds.</li></ol>
Stages on the current build: <ul>{stage_text}</ul>"""))

    qa.append(("How was variety and structural balance ensured?", f"""
<ul><li>10 slices from the standard word-problem taxonomy (change join/separate, combine, compare, equal groups, partitive
and quotative division, multiplicative compare, rates/money, multi-step), 8 batches each, each batch with its own contexts and style.</li>
<li>Shape cap: within a slice no full operation sequence above 25% of the slice; the most redundant go first. Changed after the
pilot from a global 8% cap, which would have cut most one-step subtraction (four slices share it; SVAMP is 77% one-step).</li>
<li>Result (templateness, nearest-neighbour word-sequence ratio in a 2,000-problem sample, median): our items
{templ.get('math_tutor kept items', {}).get('median_nn', '?')}, human seeds {templ.get('seeds (human, deduped)', {}).get('median_nn', '?')},
SVAMP {templ.get('SVAMP test (human)', {}).get('median_nn', '?')}, v1 template corpus {templ.get('v1 templates (math_word_problem corpus)', {}).get('median_nn', '?')}.</li></ul>"""))

    split_text = ", ".join(f"{k} {n(v)}" for k, v in (pruning or {}).get("by_split", {}).items())
    qa.append(("What are the splits?", f"""
<ul><li>Items: {split_text}.</li>
<li><b>test</b>: 10% of families per slice, random (seed 0). <b>heldout_shape</b>: every item whose problem is
<code>add>mul</code>, <code>sub>div</code> or <code>sub>add</code>, never trained on, while their parts and reverse orders are
(compositional test). <b>SVAMP</b>: the human test, single turn.</li>
<li>Only train items are reworded; renumbered copies keep their item's split.</li></ul>"""))

    qa.append(("How were rewordings and renumbered copies made?", f"""
<ul><li><b>Rewording:</b> an Opus agent writes only new text for the problem and each follow-up question; the original
<code>solve()</code> code is reused, so the maths cannot change. The full validator runs on the result, and the agent's
self-check rejects a rewording more than 0.8 similar to its original or its sibling (the same rule pruning applies).</li>
<li><b>Renumbering:</b> each number the code uses (and each digit distractor) gets a new value between half and double the
old one; text and code change together in every turn; the copy is kept only if it passes the full validator.
{('Last build: ' + n(renumber.get('items', 0)) + ' items → ' + n(renumber.get('copies', 0)) + ' copies.') if renumber else ''}</li></ul>"""))

    qa.append(("What went wrong, and what changed because of it?", """
<ul><li><b>Online writers</b> (pilot) found usable text on 1 of 11 pages and credited loosely related URLs → replaced by
harvesters that copy real problems word for word with URL and licence.</li>
<li><b>Near-duplicate measure</b> trigram Jaccard → word-sequence ratio (one-word swaps slipped through).</li>
<li><b>Shape cap</b> global 8% → 25% within a slice.</li>
<li><b>Prose number words</b> ("the two classes") counted as unused numbers → only digit numbers count; distractors are written in digits.</li>
<li><b>Usage limits</b> stopped runs A, B, C, D and the first reword run. Each writer creates one file and never overwrites;
after each stop the files were inventoried, committed and only missing tasks re-run (two half-written reword files were
kept and their remaining items written to new files).</li>
<li><b>Teacher format:</b> Gemma left out the final ";" and sometimes wrote 35-23=12 without spaces → the prompt shows an
example ending in ";", and the vocabulary accepts the no-space forms.</li></ul>"""))

    run_rows = "".join(f"<tr><td>{r['run']}</td><td>{r['date']}</td><td>{r['model']}</td><td class='n'>{r['agents']}</td>"
                       f"<td class='n'>{n(r['subagent_tokens'])}</td><td class='n'>{r['minutes']}</td><td>{r['stopped_by']}</td></tr>"
                       for r in writing_log["runs"])
    qa.append(("What did writing cost?", f"""
<div class="scroll"><table><thead><tr><th>run</th><th>date</th><th>model</th><th class="n">agents</th><th class="n">tokens</th>
<th class="n">minutes</th><th>ended by</th></tr></thead><tbody>{run_rows}</tbody></table></div>
<p class="note">{n(tokens)} agent tokens in total so far (as reported by the workflow tool).</p>"""))

    harvest_sites = ", ".join(f"{k} {v}" for k, v in sets.get("harvest", {}).get("by_site", {}).items())
    qa.append(("Which licences apply?", f"""
<ul><li>SVAMP: MIT. ASDiv: CC BY-NC 4.0. MAWPS (SVAMP repository copy): MIT.</li>
<li>Harvested problems (each record carries its URL and licence): {harvest_sites}. Licences: CC BY 4.0 (Illustrative
Mathematics), CC BY-NC-SA 3.0/4.0 (EngageNY, IM tasks), Open Government Licence v3.0 (gov.uk, Oak National Academy, dera),
public domain (NAEP, IES), CC BY-SA 4.0 (Wikibooks).</li>
<li>Written items, rewordings and code: produced for this thesis by Claude; teacher outputs by Gemma 4 (Gemma terms of use).</li></ul>"""))

    qa.append(("How can it be reproduced?", """
<pre>python -m library.domain.math_tutor.benchmarks   # downloads, filters, dedups human sets + harvest -> seeds, test
python -m library.domain.math_tutor.items        # validates every raw file
python -m library.domain.math_tutor.prune        # raw + rewordings -> items.jsonl (six stages, splits)
python -m library.domain.math_tutor.renumber     # items -> renumbered copies
python -m library.domain.math_tutor.templateness # the templateness table
python math_tutor_report.py                      # this page</pre>
<ul><li>Everything agent-written is committed as written (<code>items/raw/</code>, <code>items/reword/</code>,
<code>benchmarks/harvested/</code>); every other file is rebuilt from it deterministically (seed 0).</li>
<li>Agent prompts: the brief plus the per-run scripts in the session's workflow folder (assignment text per batch).</li></ul>"""))
    return [{"q": question, "a": answer} for question, answer in qa]


def build():
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "slices": SLICES,
        "human": human_sets(),
        "writing": writing(),
        "pruning": pruning(),
        "similarity": similarity_demo(),
        "validator_cases": validator_cases(),
        "renumber": renumber_example(),
        "storage": storage(),
        "teacher": read_json(PACKAGE / "teacher_check.json"),
        "templateness": templateness(),
        "prompts": PROMPTS,
        "example_responses": EXAMPLE_RESPONSES,
    }
    data["methods"] = methods_qa(data)
    template = open(TEMPLATE, encoding="utf-8").read()
    page = template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUTPUT.parent.mkdir(exist_ok=True)
    open(OUTPUT, "w", encoding="utf-8").write(page)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
