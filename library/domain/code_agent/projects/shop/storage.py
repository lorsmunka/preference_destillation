"""The shop's data lives in JSON files."""

import json
import os

from config import CUSTOMERS_FILE, DATA_FOLDER, ORDERS_FILE, PRODUCTS_FILE, REVIEWS_FILE


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as file:
        return json.load(file)


def write_json(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_products():
    return read_json(PRODUCTS_FILE, {})


def save_products(products):
    write_json(PRODUCTS_FILE, products)


def load_customers():
    return read_json(CUSTOMERS_FILE, {})


def save_customers(customers):
    write_json(CUSTOMERS_FILE, customers)


def load_orders():
    return read_json(ORDERS_FILE, [])


def save_orders(orders):
    write_json(ORDERS_FILE, orders)
