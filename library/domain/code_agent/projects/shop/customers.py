"""Customer accounts: sign up and log in with an email and a password."""

from storage import load_customers, save_customers


def sign_up(email, password, name):
    """Make a new account. False when the email is already used or the password is too short."""
    customers = load_customers()
    if email in customers or len(password) < 6:
        return False
    customers[email] = {"password": password, "name": name, "orders": 0}
    save_customers(customers)
    return True


def log_in(email, password):
    """The customer's name when email and password match, otherwise None."""
    customer = load_customers().get(email.lower())
    if customer and customer["password"] == password:
        return customer["name"]
    return None


def count_order(email):
    """Add one to the customer's number of orders (loyal customers get offers)."""
    customers = load_customers()
    customers[email]["orders"] += 1
    save_customers(customers)


def is_loyal(email):
    return load_customers().get(email, {}).get("orders", 0) >= 5
