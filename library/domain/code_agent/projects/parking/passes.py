"""Monthly passes: park as often as you like for one month."""

from clock import now
from config import MONTHLY_PASS_PRICE
from storage import load_passes, save_passes
from vehicles import clean_plate


def buy_pass(plate, month):
    """Sell a pass for one month ("2026-11"). The price."""
    passes = load_passes()
    passes.setdefault(clean_plate(plate), []).append(month)
    save_passes(passes)
    return MONTHLY_PASS_PRICE


def has_valid_pass(plate):
    """True when the plate has a pass for the current month."""
    return now()[:7] in load_passes().get(clean_plate(plate), [])


def months_with_pass(plate):
    return sorted(load_passes().get(clean_plate(plate), []))
