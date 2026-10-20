"""Small helpers used all over the bank simulator."""

import random
from datetime import date

from config import CURRENCY


def format_money(amount):
    """Turn 12.5 into "$12.50"."""
    return f"{CURRENCY}{amount:.2f}"


def today():
    """Today's date as text, for example "2026-10-20"."""
    return date.today().isoformat()


def current_month():
    """The month part of today's date, for example "2026-10"."""
    return today()[:7]


def new_id(prefix):
    """A random id such as "T48213" for a transaction."""
    return prefix + str(random.randint(10000, 99999))


def ask_number(question):
    """Ask until the user types a number."""
    while True:
        text = input(question)
        try:
            return float(text)
        except ValueError:
            print("Please type a number.")
