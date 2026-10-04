"""The executable side of an item. Every problem carries a solve() function; the answer is
whatever that function returns, never what anyone wrote down. Human benchmarks come with
an equation instead, which is turned into the same kind of function here.
"""

import ast
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

OPERATION_NAMES = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
    ast.Div: "div", ast.FloorDiv: "div", ast.Mod: "mod",
}


class NotAllowed(ValueError):
    """The equation or function breaks one of the domain's rules; the message says which."""


def trace_equation(equation: str) -> Tuple[List[Fraction], List[Tuple[str, Fraction]]]:
    """Evaluate an arithmetic equation exactly. Returns its numbers and every operation
    with the value it produced, in the order they are computed."""
    numbers, operations = [], []

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = Fraction(node.value).limit_denominator(1000)
            numbers.append(value)
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif right == 0:
                raise NotAllowed("division by zero")
            else:
                value = left / right
            operations.append((OPERATION_NAMES[type(node.op)], value))
            return value
        raise NotAllowed("not plain arithmetic")

    try:
        tree = ast.parse(equation.strip(), mode="eval")
    except SyntaxError:
        raise NotAllowed("not plain arithmetic")
    evaluate(tree.body)
    return numbers, operations


def equation_to_solve(equation: str) -> str:
    """'( 76.0 - 25.0 )' -> a solve() that returns {'answer': 76 - 25}. Whole numbers only,
    and division becomes // (the equation was already checked to divide exactly)."""

    class WholeNumbers(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, float):
                if not node.value.is_integer():
                    raise NotAllowed("not whole numbers")
                return ast.copy_location(ast.Constant(int(node.value)), node)
            return node

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.Div):
                node.op = ast.FloorDiv()
            return node

    tree = WholeNumbers().visit(ast.parse(equation.strip(), mode="eval"))
    return "def solve():\n    return {'answer': " + ast.unparse(tree.body) + "}"


def run_solve(source: str, state: Optional[Dict] = None) -> Dict:
    """Run a solve() function with no builtins available and return the dict it gives."""
    namespace = {"__builtins__": {}}
    exec(compile(source, "<solve>", "exec"), namespace)
    result = namespace["solve"]() if state is None else namespace["solve"](state)
    if not isinstance(result, dict) or "answer" not in result:
        raise NotAllowed("solve() must return a dict with an 'answer'")
    return result


def operation_shape(source: str) -> str:
    """The operations a function performs, in the order they are computed: 'add>mul'."""
    names = []

    def walk(node):
        if isinstance(node, ast.BinOp):
            walk(node.left)
            walk(node.right)
            names.append(OPERATION_NAMES.get(type(node.op), "other"))
        else:
            for child in ast.iter_child_nodes(node):
                walk(child)

    walk(ast.parse(source))
    return ">".join(names)
