"""Turning a cart into an order, and cancelling orders."""

from config import TAX_RATE
from discounts import apply_discount
from inventory import take_from_stock
from products import get_product
from shipping import shipping_cost
from storage import load_orders, save_orders


def order_total(cart, code="", express=False):
    """Subtotal after the discount, plus tax, plus shipping."""
    discounted = apply_discount(cart.subtotal(), code)
    tax = round(discounted * TAX_RATE, 2)
    return round(discounted + tax + shipping_cost(discounted, express), 2)


def place_order(customer_email, cart, code="", express=False):
    """Take the items from stock and save the order. The new order's number, or None."""
    if cart.is_empty():
        return None
    for product_id, quantity in cart.items.items():
        if not take_from_stock(get_product(product_id), quantity):
            return None
    orders = load_orders()
    orders.append({"number": len(orders) + 1, "customer": customer_email, "items": dict(cart.items),
                   "total": order_total(cart, code, express), "status": "paid"})
    save_orders(orders)
    return len(orders)


def cancel_order(number):
    """Cancel an order that has not been sent yet, and put its items back in stock."""
    orders = load_orders()
    order = orders[number - 1]
    if order["status"] != "paid":
        return False
    for product_id, quantity in order["items"].items():
        product = get_product(product_id)
        product.stock += quantity
        product.save()
    order["status"] = "cancelled"
    save_orders(orders)
    return True
