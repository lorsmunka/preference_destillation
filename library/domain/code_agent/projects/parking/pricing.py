"""What parking costs."""

import math

from config import DAILY_MAXIMUM, FREE_MINUTES, HOURLY_RATE, LOST_TICKET_FEE, MOTORBIKE_DISCOUNT


def parking_fee(minutes, kind="car"):
    """The first FREE_MINUTES are free; then every started hour costs HOURLY_RATE, at most
    DAILY_MAXIMUM per started day. Motorbikes pay half."""
    if minutes <= FREE_MINUTES:
        return 0.0
    days, rest = divmod(minutes, 24 * 60)
    fee = days * DAILY_MAXIMUM + min(math.ceil(rest / 60) * HOURLY_RATE, DAILY_MAXIMUM)
    if kind == "motorbike":
        fee *= MOTORBIKE_DISCOUNT
    return round(fee, 2)


def lost_ticket_fee(kind="car"):
    """A lost ticket costs a fixed amount (half for motorbikes)."""
    return LOST_TICKET_FEE * (MOTORBIKE_DISCOUNT if kind == "motorbike" else 1)


def price_table():
    """The prices shown on the sign at the entry."""
    return [(hours, parking_fee(hours * 60)) for hours in (1, 2, 3, 4, 6, 8, 12, 24)]
