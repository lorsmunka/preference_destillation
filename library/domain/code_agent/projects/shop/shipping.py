"""What delivery costs and how long it takes."""

from config import EXPRESS_EXTRA, FREE_SHIPPING_OVER, SHIPPING_COST


def shipping_cost(subtotal, express=False):
    """Free above FREE_SHIPPING_OVER; express delivery always costs extra."""
    cost = 0 if subtotal >= FREE_SHIPPING_OVER else SHIPPING_COST
    if express:
        cost += EXPRESS_EXTRA
    return round(cost, 2)


def delivery_days(country, express=False):
    """Working days until the parcel arrives."""
    days = 2 if country == "home" else 6
    if express:
        days = max(1, days - 2)
    return days


def free_shipping_gap(subtotal):
    """How much more to spend for free shipping (0 when it is already free)."""
    return max(0, round(FREE_SHIPPING_OVER - subtotal, 2))
