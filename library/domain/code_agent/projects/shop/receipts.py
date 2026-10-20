"""The receipt shown after an order."""

from config import TAX_RATE
from discounts import apply_discount
from products import get_product
from shipping import shipping_cost


def receipt_lines(cart, code=""):
    """One line per product, then the subtotal, discount, tax and shipping lines."""
    lines = []
    for product_id, quantity in cart.items.items():
        product = get_product(product_id)
        lines.append(f"{quantity} x {product.name:<20} {product.price * quantity:>8.2f}")
    subtotal = cart.subtotal()
    discounted = apply_discount(subtotal, code)
    lines.append(f"{'subtotal':<24} {subtotal:>8.2f}")
    if discounted != subtotal:
        lines.append(f"{'discount':<24} {discounted - subtotal:>8.2f}")
    lines.append(f"{'tax':<24} {discounted * TAX_RATE:>8.2f}")
    lines.append(f"{'shipping':<24} {shipping_cost(discounted):>8.2f}")
    return lines


def print_receipt(number, cart, code=""):
    print(f"Order {number}")
    for line in receipt_lines(cart, code):
        print("  " + line)
