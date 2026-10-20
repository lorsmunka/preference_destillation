"""What a stay costs."""

from config import CITY_TAX_PER_NIGHT, LONG_STAY_DISCOUNT, LONG_STAY_NIGHTS, PRICE_PER_NIGHT, WEEKEND_EXTRA
from dates import each_night, is_weekend


def night_price(kind, day):
    """One night in a room of this kind; weekend nights cost WEEKEND_EXTRA more."""
    price = PRICE_PER_NIGHT[kind]
    if is_weekend(day):
        price *= 1 + WEEKEND_EXTRA
    return price


def stay_price(kind, arrival, departure, people=1):
    """All nights, a discount for long stays, then the city tax per person and night."""
    nights = each_night(arrival, departure)
    total = sum(night_price(kind, night) for night in nights)
    if len(nights) >= LONG_STAY_NIGHTS:
        total *= 1 - LONG_STAY_DISCOUNT
    total += CITY_TAX_PER_NIGHT * people * len(nights)
    return round(total, 2)


def price_per_night(kind, arrival, departure):
    """The average price of a night, without the city tax."""
    nights = each_night(arrival, departure)
    return round(sum(night_price(kind, night) for night in nights) / len(nights), 2) if nights else 0
