"""Messages for members: overdue reminders and "your reserved book is back"."""

from dates import days_between, today
from fines import fine_for_loan
from reservations import next_in_line
from storage import load_loans


def overdue_message(loan):
    days = days_between(loan["due"], today())
    fine = fine_for_loan(loan)
    return f"Book {loan['isbn']} is {days} days late. Your fine so far: {fine:.2f}."


def send_reminders():
    """Print a reminder for every book that is still out after its due date."""
    count = 0
    for loan in load_loans():
        if loan["returned"] is None and days_between(loan["due"], today()) > 0:
            print(f"To {loan['member']}: " + overdue_message(loan))
            count += 1
    return count


def book_is_back(isbn):
    """Tell the first member in line that the reserved book can be picked up."""
    member_id = next_in_line(isbn)
    if member_id is not None:
        print(f"To {member_id}: the book {isbn} you reserved is waiting for you.")
    return member_id
