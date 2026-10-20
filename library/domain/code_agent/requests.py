"""The user's requests about a project version, each with its true places, and the item file.

A request finds or explains something, named by its meaning (a topic of the base project:
"the code that takes money out of an account") or by its name (any class, function, method or
top-level constant: "Find the withdraw method of Account."). Each version gets one request of each
of the four kinds. An item = one request on one version; the version itself is rebuilt from its
seed whenever it is needed (project.make_version), so items stay small.

Splits (decided 2026-10-20): test = whole base projects the student never sees in training
(TEST_PROJECTS), like the math tutor's held-out families.

    python -m library.domain.code_agent.requests     # items.jsonl, then a sample
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

from .project import Version, load_base, make_version, project_list, symbol_kinds

PACKAGE = Path(__file__).resolve().parent
ITEMS_PATH = PACKAGE / "items.jsonl"
TEST_PROJECTS = ("shop", "todo")   # drawn once all 8 existed: sorted(random.Random(0).sample(project_list(), 2))
VERSIONS = {"train": 500, "test": 50}   # versions per base project
KINDS = ("find_meaning", "explain_meaning", "find_name", "explain_name")   # one request of each per version

WH_WORDS = ("where", "how", "what", "which", "why", "when", "who")
FIND_NOUN = ["Find {phrase}.", "Where is {phrase}?", "Show me {phrase}.", "Which lines hold {phrase}?",
             "I'm looking for {phrase}.", "Where can I find {phrase}?"]
FIND_CLAUSE = ["Show me {phrase}.", "Find {phrase}.", "Can you find {phrase}?", "I need to see {phrase}."]
EXPLAIN = ["Explain {phrase}.", "Can you explain {phrase}?", "Tell me {phrase}.",
           "I don't get {phrase}. Can you explain it?", "Help me understand {phrase}."]
FIND_NAME = {
    "function": ["Find the {name} function.", "Where is {name} defined?",
                 "Which file has the {name} function, and on which lines?", "Show me {name}."],
    "class": ["Find the {name} class.", "Where is the {name} class?", "Which file has the {name} class, and on which lines?"],
    "method": ["Find the {method} method of {cls}.", "Where is {cls}.{method}?", "Show me the {method} method in the {cls} class."],
    "constant": ["Where is {name} set?", "Find {name}.", "Which line sets {name}?"],
}
EXPLAIN_NAME = {
    "function": ["Explain the {name} function.", "What does {name} do?", "How does {name} work?"],
    "class": ["Explain the {name} class.", "What is the {name} class for?", "How does the {name} class work?"],
    "method": ["What does {cls}.{method} do?", "Explain the {method} method of {cls}."],
}


def split_of(project: str) -> str:
    return "test" if project in TEST_PROJECTS else "train"


def meaning_request(version: Version, topic: Dict, kind: str, random_generator: random.Random) -> Dict:
    phrase = version.fill(random_generator.choice(topic["find" if kind == "find" else "explain"]))
    if kind == "explain":
        template = random_generator.choice(EXPLAIN)
    else:
        template = random_generator.choice(FIND_CLAUSE if phrase.split()[0] in WH_WORDS else FIND_NOUN)
    return {"kind": kind, "source": "meaning", "topic": topic["id"], "message": template.format(phrase=phrase),
            "truth": [list(version.place(place)) for place in topic["places"]]}


def name_request(version: Version, place, symbol_kind: str, kind: str, random_generator: random.Random) -> Dict:
    file_name, name = place
    templates = (FIND_NAME if kind == "find" else EXPLAIN_NAME)[symbol_kind]
    cls, _, method = name.partition(".")
    message = random_generator.choice(templates).format(name=name, cls=cls, method=method)
    return {"kind": kind, "source": "name", "symbol": name, "symbol_kind": symbol_kind, "message": message,
            "truth": [[file_name, name]]}


def version_requests(base: Dict, version: Version) -> List[Dict]:
    """One request of each kind (KINDS) on this version: two different topics, two different names."""
    random_generator = random.Random(f"{version.id}-requests")
    topics = random_generator.sample(base["topics"], 2)
    kinds = symbol_kinds(version.files)
    findable = sorted(place for place, symbol_kind in kinds.items() if not place[1].split(".")[-1].startswith("__"))
    explainable = [place for place in findable if kinds[place] != "constant"]
    find_place = random_generator.choice(findable)
    explain_place = random_generator.choice([place for place in explainable if place != find_place])
    return [meaning_request(version, topics[0], "find", random_generator),
            meaning_request(version, topics[1], "explain", random_generator),
            name_request(version, find_place, kinds[find_place], "find", random_generator),
            name_request(version, explain_place, kinds[explain_place], "explain", random_generator)]


def build_items() -> List[Dict]:
    """Every item of every base project, written to ITEMS_PATH (a derived file, rebuilt each time)."""
    items = []
    for project in project_list():
        base, split = load_base(project), split_of(project)
        for number in range(VERSIONS[split]):
            version = make_version(base, number)
            for index, request in enumerate(version_requests(base, version)):
                items.append({"id": f"{version.id}#{index}", "project": project, "version": number, "split": split, **request})
    with open(ITEMS_PATH, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return items


def read_items() -> List[Dict]:
    return [json.loads(line) for line in open(ITEMS_PATH, encoding="utf-8")]


def main() -> None:
    from collections import Counter
    items = build_items()
    print(f"{len(items)} items -> {ITEMS_PATH}")
    print(dict(Counter((item["split"], item["project"]) for item in items)))
    print(dict(Counter((item["kind"], item["source"]) for item in items)))
    for item in random.Random(0).sample(items, 12):
        print(f"  {item['id']:<22} {item['message']:<70} {item['truth']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
