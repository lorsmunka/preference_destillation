"""The shopping cart: what the customer wants to buy, before ordering."""

from config import MAX_PER_ITEM
from products import get_product


class Cart:
    def __init__(self):
        self.items = {}   # product id -> quantity

    def add_item(self, product_id, quantity=1):
        """None when the item went in, otherwise why not."""
        product = get_product(product_id)
        if product is None:
            return "No such product."
        new_quantity = self.items.get(product_id, 0) + quantity
        if new_quantity > MAX_PER_ITEM:
            return f"You can buy at most {MAX_PER_ITEM} of one item."
        if not product.in_stock(new_quantity):
            return f"Only {product.stock} left."
        self.items[product_id] = new_quantity
        return None

    def remove_item(self, product_id):
        self.items.pop(product_id, None)

    def subtotal(self):
        """The price of everything in the cart, before tax and shipping."""
        return round(sum(get_product(product_id).price * quantity for product_id, quantity in self.items.items()), 2)

    def is_empty(self):
        return not self.items
