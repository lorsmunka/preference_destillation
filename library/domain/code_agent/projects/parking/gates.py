"""The entry and exit barriers."""

from clock import minutes_between, now
from config import FREE_MINUTES
from spots import leave
from tickets import issue_ticket

EXIT_MINUTES = 15   # after paying, you have this long to drive out


def open_entry_gate(vehicle):
    """Print a ticket and lift the barrier, or keep it down when the garage is full."""
    ticket = issue_ticket(vehicle)
    if ticket is None:
        print("FULL - please come back later.")
        return None
    print(f"Ticket {ticket.number}: park at {ticket.spot}.")
    return ticket


def may_leave(ticket):
    """A paid ticket opens the exit; so does a short visit inside the free minutes."""
    return ticket.paid or minutes_between(ticket.entered, now()) <= FREE_MINUTES


def open_exit_gate(ticket):
    if not may_leave(ticket):
        print("Please pay at the machine first.")
        return False
    leave(ticket.spot)
    print("Goodbye!")
    return True
