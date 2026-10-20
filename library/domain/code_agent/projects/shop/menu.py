"""The shop's text menu."""

MENU = [
    ("b", "Browse a category"),
    ("a", "Add to cart"),
    ("r", "Remove from cart"),
    ("c", "Check out"),
    ("x", "Cancel an order"),
    ("q", "Leave the shop"),
]


def print_menu(cart_size):
    print(f"\nCart: {cart_size} item(s)")
    for key, label in MENU:
        print(f"  [{key}] {label}")


def read_choice():
    keys = [key for key, _ in MENU]
    while True:
        choice = input("> ").strip().lower()
        if choice in keys:
            return choice
        print("Choose " + "/".join(keys))
