"""Savings accounts earn interest once a month."""

from config import SAVINGS_INTEREST_RATE
from transactions import record_transaction


def monthly_interest(balance):
    """One month of interest: the yearly rate split over 12 months, rounded to cents."""
    if balance <= 0:
        return 0
    return round(balance * SAVINGS_INTEREST_RATE / 12, 2)


def add_interest(account):
    """Add this month's interest to a savings Account."""
    if account.kind != "savings":
        return 0
    interest = monthly_interest(account.balance)
    if interest > 0:
        account.balance += interest
        record_transaction(account.number, "interest", interest)
        account.save()
    return interest
