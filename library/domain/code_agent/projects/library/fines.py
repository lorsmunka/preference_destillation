"""Late fees: a small fine for every day a book comes back after its due date."""

from config import LATE_FEE_PER_DAY, MAX_LATE_FEE
from dates import days_between, today
from storage import load_loans


def late_fee(days_late):
    """The fine for a book that is `days_late` days late, never more than MAX_LATE_FEE."""
    if days_late <= 0:
        return 0
    return min(days_late * LATE_FEE_PER_DAY, MAX_LATE_FEE)


def fine_for_loan(loan):
    """The fine of one loan: up to the day it came back, or up to today if it is still out."""
    end = loan["returned"] or today()
    return late_fee(days_between(loan["due"], end))


def unpaid_fines(member_id):
    """All fines of a member's loans that have not been paid yet."""
    total = 0
    for loan in load_loans():
        if loan["member"] == member_id and not loan.get("fine_paid"):
            total += fine_for_loan(loan)
    return round(total, 2)
