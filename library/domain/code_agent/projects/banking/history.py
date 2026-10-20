"""Showing an account's past transactions."""

from transactions import transactions_of
from utils import format_money


def transactions_of_kind(account_number, kind):
    return [t for t in transactions_of(account_number) if t["kind"] == kind]


def last_transactions(account_number, count=5):
    """The newest `count` transactions, newest first."""
    return list(reversed(transactions_of(account_number)))[:count]


def print_statement(account_number):
    """Print every transaction with a + or - sign and the balance after it."""
    running = 0
    print(f"Statement for account {account_number}")
    print("date        kind           amount     balance")
    for t in transactions_of(account_number):
        if t["kind"] in ("deposit", "interest", "transfer_in"):
            running += t["amount"]
            sign = "+"
        else:
            running -= t["amount"]
            sign = "-"
        print(f"{t['date']}  {t['kind']:<13} {sign}{format_money(t['amount']):>9}  {format_money(running):>10}")
