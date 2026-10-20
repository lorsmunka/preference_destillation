"""Keeping track of how many of each product are in the warehouse."""

from config import LOW_STOCK
from products import get_product
from storage import load_products


def restock(product_id, quantity):
    """Add new items from a delivery. The new stock, or None for an unknown product."""
    product = get_product(product_id)
    if product is None:
        return None
    product.stock += quantity
    product.save()
    return product.stock


def take_from_stock(product, quantity):
    """Remove sold items. False when there are not enough."""
    if not product.in_stock(quantity):
        return False
    product.stock -= quantity
    product.save()
    return True


def low_stock_items():
    """Names of the products that are running out, fewest first."""
    low = [(data["stock"], data["name"]) for data in load_products().values() if data["stock"] <= LOW_STOCK]
    return [name for _, name in sorted(low)]
