"""Discount codes customers can type in at checkout."""

CODES = {
    "WELCOME10": ("percent", 10),
    "SUMMER20": ("percent", 20),
    "FIVEOFF": ("amount", 5),
}
MIN_ORDER_FOR_AMOUNT = 25


def code_is_valid(code):
    return code.upper() in CODES


def apply_discount(subtotal, code):
    """The subtotal after a discount code. An amount code only works on orders of at least 25."""
    if not code_is_valid(code):
        return subtotal
    kind, value = CODES[code.upper()]
    if kind == "percent":
        return round(subtotal * (100 - value) / 100, 2)
    if subtotal >= MIN_ORDER_FOR_AMOUNT:
        return round(subtotal - value, 2)
    return subtotal


def saving(subtotal, code):
    """How much the code saves on this subtotal."""
    return round(subtotal - apply_discount(subtotal, code), 2)
