"""Checks game files: the app's compiler, then what it lets through (keyword in the wrong block, wrong argument
count, bad colour, unknown id), then the signs of a hollow game. Prints one line per file; exit code 1 when any
file fails.

    python -m library.domain.game_dsl.check path/to/game.dsl [more.dsl ...]
"""

import sys
from pathlib import Path

from .compiler import compile_programs
from .language import defined_ids, hollow, parse_program, problems
from .manual import load_reference


def check_sources(sources):
    """{name: source} -> {name: list of problems (empty = clean)}"""
    reference = load_reference()
    compiled = compile_programs(sources)
    results = {}
    for name, source in sources.items():
        found = []
        compiles, error = compiled[name]
        if not compiles:
            found.append(f"does not compile: {error}")
        try:
            root = parse_program(source)
            for kind, items in problems([root], None, reference, defined_ids(root)).items():
                found += [f"{kind}: {item}" for item in items]
            found += [f"hollow: {item}" for item in hollow(root)]
        except ValueError as parse_error:
            found.append(f"does not parse: {parse_error}")
        results[name] = found
    return results


def main():
    paths = [Path(argument) for argument in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(2)
    results = check_sources({str(path): path.read_text(encoding="utf-8") for path in paths})
    failed = 0
    for name, found in results.items():
        lines = Path(name).read_text(encoding="utf-8").count("\n") + 1
        if found:
            failed += 1
            print(f"FAIL {name} ({lines} lines)")
            for item in found:
                print(f"     - {item}")
        else:
            print(f"OK   {name} ({lines} lines)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
