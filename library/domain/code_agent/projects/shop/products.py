"""The things the shop sells."""

from storage import load_products, save_products


class Product:
    def __init__(self, product_id, name, price, stock, category):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock = stock
        self.category = category

    def in_stock(self, quantity=1):
        return self.stock >= quantity

    def save(self):
        products = load_products()
        products[self.product_id] = {"name": self.name, "price": self.price,
                                     "stock": self.stock, "category": self.category}
        save_products(products)


def get_product(product_id):
    data = load_products().get(product_id)
    if data is None:
        return None
    return Product(product_id, data["name"], data["price"], data["stock"], data["category"])


def products_in(category):
    """Every product of one category, cheapest first."""
    found = [get_product(product_id) for product_id, data in load_products().items() if data["category"] == category]
    return sorted(found, key=lambda product: product.price)
