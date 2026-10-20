"""Paying at the machine with coins and notes."""

from clock import minutes_between, now
from passes import has_valid_pass
from pricing import parking_fee
from tickets import mark_paid

COINS = [20, 10, 5, 2, 1, 0.5, 0.2, 0.1]


def amount_due(ticket):
    """What this ticket costs now; nothing for a vehicle with a monthly pass."""
    if has_valid_pass(ticket.plate):
        return 0.0
    return parking_fee(minutes_between(ticket.entered, now()), ticket.kind)


def change_coins(paid, due):
    """The change as a list of coins and notes, the biggest first."""
    change = round(paid - due, 2)
    coins = []
    for coin in COINS:
        while change >= coin - 0.001:
            coins.append(coin)
            change = round(change - coin, 2)
    return coins


def pay(ticket, paid):
    """Pay a ticket. The change, or None when not enough money was put in."""
    due = amount_due(ticket)
    if paid < due:
        return None
    mark_paid(ticket)
    return change_coins(paid, due)
