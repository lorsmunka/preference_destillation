"""Fees the bank charges: a monthly fee for small balances and a fee for going below zero."""

from config import FREE_BALANCE, MONTHLY_FEE, OVERDRAFT_FEE
from transactions import record_transaction


def overdraft_fee(balance):
    """The fee for a balance below zero: a fixed fee, plus 1 for every full 50 below zero."""
    if balance >= 0:
        return 0
    return OVERDRAFT_FEE + int(-balance // 50)


def monthly_fee(balance):
    """Accounts with at least FREE_BALANCE pay nothing; the others pay MONTHLY_FEE."""
    if balance >= FREE_BALANCE:
        return 0
    return MONTHLY_FEE


def charge_monthly_fee(account):
    """Take this month's fee from an Account, if it has to pay one."""
    fee = monthly_fee(account.balance)
    if fee > 0:
        account.balance -= fee
        record_transaction(account.number, "fee", fee, "monthly")
        account.save()
    return fee
