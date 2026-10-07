import sys
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.functions import execute, NotAllowed, equation_to_solve, trace_equation
from library.domain.math_tutor.items import validate, text_numbers
cases = [
    ("first-turn", "def solve():\n    x = 5\n    x += 3\n    return {'answer': x}", None),
    ("answer before **state", "def solve(state):\n    y = state['a'] + 1\n    return {'answer': y, **state}", {"a": 3, "answer": 3}),
    ("no answer key, inherits", "def solve(state):\n    y = state['a'] + 1\n    return {**state, 'y': y}", {"a": 3, "answer": 3}),
    ("walrus", "def solve():\n    return {'answer': (x := 5) + 1}", None),
    ("bool literal", "def solve():\n    return {'answer': True + 1}", None),
    ("unary minus", "def solve():\n    return {'answer': -5 + 9}", None),
    ("pow", "def solve():\n    return {'answer': 2 ** 3}", None),
    ("docstring", "def solve():\n    '''doc'''\n    return {'answer': 2 + 3}", None),
    ("nested def", "def solve():\n    def f():\n        return 1\n    return {'answer': 2 + 3}", None),
    ("tuple assign", "def solve():\n    a, b = 1, 2\n    return {'answer': a + b}", None),
    ("chained assign", "def solve():\n    a = b = 2\n    return {'answer': a + b}", None),
    ("state assign", "def solve(state):\n    state['a'] = 2\n    return {'answer': state['a'] + 1}", {"a": 1}),
    ("intermediate 1000 then back", "def solve():\n    a = 40 * 40 // 40\n    return {'answer': a}", None),
    ("return mid", "def solve():\n    return {'answer': 1 + 1}\n    return {'answer': 2}", None),
    ("state used as name", "def solve(state):\n    return {'answer': state + 1}", {"a": 1}),
    ("dict comprehension", "def solve():\n    return {k: 1 for k in 'ab'}", None),
    ("async def", "async def solve():\n    return {'answer': 1 + 1}", None),
    ("lambda default", "def solve(x=1):\n    return {'answer': 1 + 1}", None),
    ("type comment / annotation", "def solve() -> int:\n    return {'answer': 1 + 1}", None),
    ("annassign", "def solve():\n    x: int = 3\n    return {'answer': x + 1}", None),
    ("subscript string key on non-state", "def solve():\n    a = 3\n    return {'answer': a['x']}", None),
    ("null byte", "def solve():\n    return {'answer': 1 + 1}\x00", None),
    ("deep nesting", "def solve():\n    return {'answer': " + "(" * 300 + "1" + ")" * 300 + " + 1}", None),
    ("long chain", "def solve():\n    return {'answer': " + " + ".join(["0"] * 20000) + "}", None),
]
for name, source, state in cases:
    try:
        e = execute(source, state)
        print(f"OK     {name:32} result={e.result} ops={e.operations} reads_state={e.reads_state}")
    except NotAllowed as error:
        print(f"REFUSE {name:32} {error}")
    except BaseException as error:
        print(f"CRASH  {name:32} {type(error).__name__}: {str(error)[:80]}")
print(trace_equation("( 76.0 - 25.0 )"), equation_to_solve("( 76.0 - 25.0 )"))
print(text_numbers("a ten-dollar note, twenty one, Twenty-Two, seventy-teen, nineteen-ninety"))
