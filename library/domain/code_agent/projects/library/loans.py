"""Borrowing and returning books. A loan is saved as a small dictionary in loans.json."""

from config import LOAN_DAYS, MAX_RENEWALS, RENEW_DAYS
from dates import add_days, today
from storage import load_loans, save_loans


def loans_of(member_id):
    """The books this member has at home right now."""
    return [loan for loan in load_loans() if loan["member"] == member_id and loan["returned"] is None]


def borrow_book(member, book):
    """Lend `book` to `member`. None when it worked, otherwise the reason it did not."""
    problem = member.can_borrow()
    if problem:
        return problem
    if not book.take_copy():
        return "No copy is on the shelf. You can reserve it."
    loans = load_loans()
    loans.append({"member": member.member_id, "isbn": book.isbn, "borrowed": today(),
                  "due": add_days(today(), LOAN_DAYS), "renewals": 0, "returned": None})
    save_loans(loans)
    book.save()
    return None


def return_book(member_id, book):
    """Mark the member's loan of this book as returned and put the copy back on the shelf."""
    loans = load_loans()
    for loan in loans:
        if loan["member"] == member_id and loan["isbn"] == book.isbn and loan["returned"] is None:
            loan["returned"] = today()
            save_loans(loans)
            book.put_back()
            book.save()
            return loan
    return None


def renew_loan(loan):
    """Give a loan more days, at most MAX_RENEWALS times."""
    if loan["renewals"] >= MAX_RENEWALS:
        return False
    loan["due"] = add_days(loan["due"], RENEW_DAYS)
    loan["renewals"] += 1
    return True
