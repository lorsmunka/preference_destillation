"""A Ticket is printed at the entry gate and paid before leaving."""

from clock import now
from spots import find_free_spot, park
from storage import load_tickets, save_tickets


class Ticket:
    def __init__(self, number, plate, kind, spot, entered, paid=False):
        self.number = number
        self.plate = plate
        self.kind = kind
        self.spot = spot
        self.entered = entered
        self.paid = paid

    def as_data(self):
        return {"plate": self.plate, "kind": self.kind, "spot": self.spot, "entered": self.entered, "paid": self.paid}


def issue_ticket(vehicle):
    """Give the vehicle a free spot and a ticket. None when the garage is full."""
    spot = find_free_spot(vehicle.kind)
    if spot is None:
        return None
    tickets = load_tickets()
    number = str(len(tickets) + 1001)
    ticket = Ticket(number, vehicle.plate, vehicle.kind, spot, now())
    tickets[number] = ticket.as_data()
    save_tickets(tickets)
    park(spot, vehicle.plate)
    return ticket


def find_ticket(number):
    data = load_tickets().get(number)
    if data is None:
        return None
    return Ticket(number, data["plate"], data["kind"], data["spot"], data["entered"], data["paid"])


def mark_paid(ticket):
    tickets = load_tickets()
    tickets[ticket.number]["paid"] = True
    save_tickets(tickets)
