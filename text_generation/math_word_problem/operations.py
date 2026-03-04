import random


ARITHMETIC_OPERATIONS = [
    "add", "sub", "mul", "div",
    "mul_add", "mul_sub", "add_mul", "sub_mul", "add_sub", "div_add",
]

COMPARISON_OPERATIONS = ["cmp", "cmp_sub", "cmp_mul"]

DISTRIBUTION = [
    (20, 1, 9, False, False),
    (15, 10, 99, True, False),
    (10, 10, 99, False, False),
    (15, 100, 999, True, False),
    (10, 100, 999, False, False),
    (15, 10, 99, True, True),
    (15, 10, 99, False, True),
]

DISTRIBUTION_WEIGHTS = [entry[0] for entry in DISTRIBUTION]


def pick_distribution_bucket():
    return random.choices(DISTRIBUTION, weights=DISTRIBUTION_WEIGHTS, k=1)[0]


def generate_number_in_range(low, high, use_round):
    if use_round:
        if high < 100:
            step = 10
        else:
            step = 100
        low_rounded = ((low + step - 1) // step) * step
        high_rounded = (high // step) * step
        if low_rounded > high_rounded:
            return random.randint(low, high)
        return random.randrange(low_rounded, high_rounded + 1, step)
    return random.randint(low, high)


def generate_numbers(operation, range_low, range_high, use_round):
    def generator(): return generate_number_in_range(range_low, range_high, use_round)

    if operation == "div":
        return generate_division_numbers(generator, range_low, range_high, use_round)
    if operation == "div_add":
        values = generate_division_numbers(
            generator, range_low, range_high, use_round)
        values["C"] = generator()
        return values

    values = {}
    for key in get_variable_keys(operation):
        values[key] = generator()

    if operation in ("sub", "sub_mul"):
        if values["A"] <= values["B"]:
            values["A"], values["B"] = max(
                values["A"], values["B"]), min(values["A"], values["B"])
            if values["A"] == values["B"]:
                values["A"] += 1

    if operation == "mul_sub":
        product = values["A"] * values["B"]
        if values["C"] >= product:
            values["C"] = random.randint(1, max(1, product - 1))

    if operation == "add_sub":
        total = values["A"] + values["B"]
        if values["C"] >= total:
            values["C"] = random.randint(1, max(1, total - 1))

    return values


def generate_division_numbers(generator, range_low, range_high, use_round):
    divisor = generator()
    if divisor == 0:
        divisor = 1
    max_quotient = range_high // divisor
    min_quotient = max(1, range_low // divisor)
    quotient = generate_number_in_range(
        min_quotient, max(min_quotient, max_quotient), use_round)
    dividend = divisor * quotient
    return {"A": dividend, "B": divisor}


def get_variable_keys(operation):
    if operation in ("add", "sub", "mul", "div", "cmp"):
        return ["A", "B"]
    return ["A", "B", "C"]


SCAFFOLDS = {
    "add": "A=?\nB=?\nC=A+B=?\nSolution: ?",
    "sub": "A=?\nB=?\nC=A-B=?\nSolution: ?",
    "mul": "A=?\nB=?\nC=A*B=?\nSolution: ?",
    "div": "A=?\nB=?\nC=A/B=?\nSolution: ?",
    "mul_add": "A=?\nB=?\nC=?\nD=A*B=?\nE=D+C=?\nSolution: ?",
    "mul_sub": "A=?\nB=?\nC=?\nD=A*B=?\nE=D-C=?\nSolution: ?",
    "add_mul": "A=?\nB=?\nC=?\nD=A+B=?\nE=D*C=?\nSolution: ?",
    "sub_mul": "A=?\nB=?\nC=?\nD=A-B=?\nE=D*C=?\nSolution: ?",
    "add_sub": "A=?\nB=?\nC=?\nD=A+B=?\nE=D-C=?\nSolution: ?",
    "div_add": "A=?\nB=?\nC=?\nD=A/B=?\nE=D+C=?\nSolution: ?",
    "cmp": "A=?\nB=?\nC=A>B=?\nSolution: ?",
    "cmp_sub": "A=?\nB=?\nC=?\nD=A-B=?\nE=D>C=?\nSolution: ?",
    "cmp_mul": "A=?\nB=?\nC=?\nD=A*B=?\nE=D>C=?\nSolution: ?",
}

# For hundreds/thousands ranges, avoid multiplication-heavy operations
# (multiplying two 3-digit numbers gives unreasonably large results)
LARGE_RANGE_ARITHMETIC = ["add", "sub", "div", "add_sub", "div_add"]
