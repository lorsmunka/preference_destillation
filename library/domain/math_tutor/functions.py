"""The executable side of an item. Every problem carries a solve() function; the answer is
whatever that function returns, never what anyone wrote down. Human benchmarks come with
an equation instead, which is turned into the same kind of function here.

solve() is never handed to Python's exec. A small interpreter walks its syntax tree and
only knows straight-line arithmetic: assignments, + - * // %, whole-number literals,
reading state['name'], and one return of a dict. Anything else (calls, loops, imports,
attributes) is refused, so agent-written code cannot hang or touch anything, and every
intermediate value is checked on the way.
"""

import ast
import copy
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Tuple

OPERATION_NAMES = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.FloorDiv: "div", ast.Mod: "mod"}
MAX_VALUE = 999


class NotAllowed(ValueError):
    """The equation or function breaks one of the domain's rules; the message says which."""


class Step(NamedTuple):
    """One line of solve(), as the annotations render it."""
    name: str        # what it sets: a variable, or a key computed inside the returned dict
    code: str        # the expression as written, state['x'] shown as plain x: "cost + 5"
    filled: str      # the same expression with every name replaced by its value: "12 + 5"
    value: int
    computes: bool   # False for a plain fact ("trays = 4") or a copy ("x = state['y']")


class Execution(NamedTuple):
    result: Dict[str, int]   # the dict solve() returned, "answer" included
    operations: List[str]    # every operation in the order computed: ["mul", "sub"]
    literals: List[int]      # every number written in the code
    reads_state: bool        # whether it used anything from the previous turn
    steps: List[Step]        # every assignment (and computed return value), in order


def parse_solve(source: str, takes_state: bool) -> ast.FunctionDef:
    try:
        module = ast.parse(source)
    except SyntaxError:
        raise NotAllowed("syntax error")
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef) or module.body[0].name != "solve":
        raise NotAllowed("must be exactly one def solve")
    function = module.body[0]
    arguments = function.args
    expected = ["state"] if takes_state else []
    if ([argument.arg for argument in arguments.args] != expected or arguments.vararg or arguments.kwarg
            or arguments.kwonlyargs or arguments.posonlyargs or arguments.defaults or function.decorator_list):
        raise NotAllowed(f"solve must take ({', '.join(expected)})")
    if not function.body or not isinstance(function.body[-1], ast.Return):
        raise NotAllowed("solve must end with return")
    return function


class Interpreter:
    def __init__(self, state: Optional[Dict[str, int]], allow_remainders: bool, trace: bool = False):
        self.state = state
        self.trace = trace   # record Steps for the annotations (off for plain validation: it is slower)
        self.allow_remainders = allow_remainders
        self.variables: Dict[str, int] = {}
        self.operations: List[str] = []
        self.literals: List[int] = []
        self.reads_state = False
        self.steps: List[Step] = []

    def run(self, function: ast.FunctionDef) -> Dict[str, int]:
        for statement in function.body[:-1]:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name, expression = statement.targets[0].id, statement.value
                filled = self.filled(expression) if self.trace else ""
                self.assign(name, self.evaluate(expression))
                self.record(name, expression, filled)
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                name = statement.target.id
                current = self.read_variable(name)
                expression = ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()), op=statement.op, right=statement.value)
                filled = self.filled(expression) if self.trace else ""
                self.assign(name, self.operate(statement.op, current, self.evaluate(statement.value)))
                self.record(name, expression, filled)
            else:
                raise NotAllowed(f"{type(statement).__name__} statement not allowed")
        return self.returned_dict(function.body[-1].value)

    # --- the step trace, for annotations; it never changes what is computed --------------
    def record(self, name: str, expression, filled: str) -> None:
        if not self.trace:
            return
        computes = any(isinstance(node, ast.BinOp) for node in ast.walk(expression))
        self.steps.append(Step(name, self.as_code(expression), filled, self.variables[name], computes))

    @staticmethod
    def as_code(expression) -> str:
        """The expression with state['x'] written as x (the previous turn's quantity by name)."""
        class PlainNames(ast.NodeTransformer):
            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "state" and isinstance(node.slice, ast.Constant):
                    return ast.copy_location(ast.Name(id=node.slice.value, ctx=ast.Load()), node)
                return self.generic_visit(node)
        return ast.unparse(PlainNames().visit(copy.deepcopy(expression)))

    def filled(self, expression) -> str:
        """The expression with every name and state['x'] replaced by its current value."""
        interpreter = self

        class Values(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in interpreter.variables:
                    return ast.copy_location(ast.Constant(interpreter.variables[node.id]), node)
                return node

            def visit_Subscript(self, node):
                if (isinstance(node.value, ast.Name) and node.value.id == "state" and interpreter.state is not None
                        and isinstance(node.slice, ast.Constant) and node.slice.value in interpreter.state):
                    return ast.copy_location(ast.Constant(interpreter.state[node.slice.value]), node)
                return self.generic_visit(node)
        return ast.unparse(Values().visit(copy.deepcopy(expression)))

    def assign(self, name: str, value: int) -> None:
        if name == "state":
            raise NotAllowed("cannot assign to state")
        self.variables[name] = value

    def read_variable(self, name: str) -> int:
        if name not in self.variables:
            raise NotAllowed(f"'{name}' used before it is set")
        return self.variables[name]

    def returned_dict(self, node) -> Dict[str, int]:
        if not isinstance(node, ast.Dict):
            raise NotAllowed("solve must return a dict")
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:  # {**state, ...}: carry the previous turn's values forward
                if not (isinstance(value, ast.Name) and value.id == "state" and self.state is not None):
                    raise NotAllowed("only **state may be unpacked")
                result.update(self.state)
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                filled = self.filled(value) if self.trace else ""
                result[key.value] = self.evaluate(value)
                if self.trace and any(isinstance(node, ast.BinOp) for node in ast.walk(value)):  # computed in the return itself
                    self.steps.append(Step(key.value, self.as_code(value), filled, result[key.value], True))
            else:
                raise NotAllowed("dict keys must be plain strings")
        if "answer" not in result:
            raise NotAllowed("returned dict has no 'answer'")
        return result

    def evaluate(self, node) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            self.literals.append(node.value)
            return self.checked(node.value)
        if isinstance(node, ast.Name):
            return self.read_variable(node.id)
        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "state" and self.state is not None
                    and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
                raise NotAllowed("only state['name'] may be indexed")
            if node.slice.value not in self.state:
                raise NotAllowed(f"state has no '{node.slice.value}'")
            self.reads_state = True
            return self.state[node.slice.value]
        if isinstance(node, ast.BinOp):
            return self.operate(node.op, self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, ast.Constant):
            raise NotAllowed("only whole-number literals")
        raise NotAllowed(f"{type(node).__name__} not allowed")

    def operate(self, operator, left: int, right: int) -> int:
        if type(operator) not in OPERATION_NAMES:
            raise NotAllowed(f"operator {type(operator).__name__} not allowed")
        if isinstance(operator, (ast.FloorDiv, ast.Mod)) and right == 0:
            raise NotAllowed("division by zero")
        if isinstance(operator, ast.Mod) and not self.allow_remainders:
            raise NotAllowed("% only in remainder problems (S07)")
        if isinstance(operator, ast.FloorDiv) and left % right and not self.allow_remainders:
            raise NotAllowed("division leaves a remainder")
        value = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right,
                 ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right}[type(operator)]()
        self.operations.append(OPERATION_NAMES[type(operator)])
        return self.checked(value)

    @staticmethod
    def checked(value: int) -> int:
        if value < 0:
            raise NotAllowed("negative value")
        if value > MAX_VALUE:
            raise NotAllowed("value above 999")
        return value


def execute(source: str, state: Optional[Dict[str, int]] = None, allow_remainders: bool = False,
            trace: bool = False) -> Execution:
    """Run one solve() (state=None for the first turn, the previous result for a follow-up)."""
    interpreter = Interpreter(state, allow_remainders, trace)
    result = interpreter.run(parse_solve(source, takes_state=state is not None))
    return Execution(result, interpreter.operations, interpreter.literals, interpreter.reads_state, interpreter.steps)


# --- human benchmarks: dataset equation -> solve() -------------------------------------

def trace_equation(equation: str) -> Tuple[List[Fraction], List[Tuple[str, Fraction]]]:
    """Evaluate a dataset's arithmetic equation exactly (it may hold floats and /). Returns
    its numbers and every operation with the value it produced, in the order computed."""
    numbers, operations = [], []

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = Fraction(node.value).limit_denominator(1000)
            numbers.append(value)
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                value, name = left + right, "add"
            elif isinstance(node.op, ast.Sub):
                value, name = left - right, "sub"
            elif isinstance(node.op, ast.Mult):
                value, name = left * right, "mul"
            elif right == 0:
                raise NotAllowed("division by zero")
            else:
                value, name = left / right, "div"
            operations.append((name, value))
            return value
        raise NotAllowed("not plain arithmetic")

    try:
        tree = ast.parse(equation.strip(), mode="eval")
    except SyntaxError:
        raise NotAllowed("not plain arithmetic")
    walk(tree.body)
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
