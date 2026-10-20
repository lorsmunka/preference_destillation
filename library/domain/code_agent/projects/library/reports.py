"""Reports for the librarian."""

from collections import Counter

from dates import days_between, today
from storage import load_books, load_loans


def overdue_loans():
    """Every loan still out after its due date, the longest overdue first."""
    late = [loan for loan in load_loans() if loan["returned"] is None and days_between(loan["due"], today()) > 0]
    return sorted(late, key=lambda loan: loan["due"])


def most_borrowed(count=5):
    """The `count` books borrowed most often, with how many times."""
    times = Counter(loan["isbn"] for loan in load_loans())
    books = load_books()
    return [(books[isbn]["title"], number) for isbn, number in times.most_common(count) if isbn in books]


def shelf_report():
    """How many copies of each book are on the shelf, out of how many the library owns."""
    for isbn, data in sorted(load_books().items(), key=lambda pair: pair[1]["title"]):
        print(f"{data['title']:<30} {data['on_shelf']}/{data['copies']} on the shelf")
