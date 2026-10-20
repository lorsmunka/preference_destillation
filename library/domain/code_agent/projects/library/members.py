"""Library members: pupils and teachers who can borrow books."""

from config import FINE_LIMIT, MAX_LOANS
from fines import unpaid_fines
from loans import loans_of
from storage import load_members


class Member:
    def __init__(self, member_id, name, role="pupil"):
        self.member_id = member_id
        self.name = name
        self.role = role   # "pupil" or "teacher"

    def loan_limit(self):
        """Teachers may borrow twice as many books as pupils."""
        if self.role == "teacher":
            return MAX_LOANS * 2
        return MAX_LOANS

    def can_borrow(self):
        """None when the member may borrow another book, otherwise the reason not."""
        if len(loans_of(self.member_id)) >= self.loan_limit():
            return "You already have as many books as you may borrow."
        if unpaid_fines(self.member_id) > FINE_LIMIT:
            return "Please pay your fines first."
        return None


def find_member(member_id):
    data = load_members().get(member_id)
    if data is None:
        return None
    return Member(member_id, data["name"], data["role"])
