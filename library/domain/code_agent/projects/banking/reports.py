"""A monthly summary: how much came in, how much went out, and the fees paid."""

from transactions import transactions_of
from utils import current_month, format_money


def month_transactions(account_number, month):
    """The transactions of one month, for example month "2026-10"."""
    return [t for t in transactions_of(account_number) if t["date"].startswith(month)]


def month_summary(account_number, month=None):
    month = month or current_month()
    money_in = money_out = fees = 0
    for t in month_transactions(account_number, month):
        if t["kind"] in ("deposit", "interest", "transfer_in"):
            money_in += t["amount"]
        elif t["kind"] == "fee":
            fees += t["amount"]
        else:
            money_out += t["amount"]
    return {"month": month, "in": money_in, "out": money_out, "fees": fees}


def print_month_summary(account_number):
    summary = month_summary(account_number)
    print(f"Summary for {summary['month']}")
    print(f"  money in:  {format_money(summary['in'])}")
    print(f"  money out: {format_money(summary['out'])}")
    print(f"  fees:      {format_money(summary['fees'])}")
