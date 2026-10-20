"""Checks that an amount or an account number makes sense before money moves."""

from config import MAX_DEPOSIT, MAX_WITHDRAWAL
from storage import load_accounts


def check_amount(amount):
    """None when the amount is fine, otherwise the reason it is not."""
    if amount <= 0:
        return "The amount must be more than zero."
    if round(amount, 2) != amount:
        return "The amount can have at most two decimals."
    return None


def check_withdrawal(amount):
    problem = check_amount(amount)
    if problem:
        return problem
    if amount > MAX_WITHDRAWAL:
        return f"You can take out at most {MAX_WITHDRAWAL} at once."
    return None


def check_deposit(amount):
    problem = check_amount(amount)
    if problem:
        return problem
    if amount > MAX_DEPOSIT:
        return f"You can put in at most {MAX_DEPOSIT} at once."
    return None


def account_exists(account_number):
    return account_number in load_accounts()
