"""The text menu the user picks from after logging in."""

MENU_OPTIONS = [
    ("1", "Show balance"),
    ("2", "Deposit money"),
    ("3", "Withdraw money"),
    ("4", "Send money to another account"),
    ("5", "Show statement"),
    ("6", "Show this month's summary"),
    ("0", "Log out"),
]


def show_menu():
    print()
    for key, label in MENU_OPTIONS:
        print(f"  {key}. {label}")


def ask_choice():
    """Ask until the user types one of the menu keys."""
    keys = [key for key, _ in MENU_OPTIONS]
    while True:
        choice = input("Choose: ").strip()
        if choice in keys:
            return choice
        print("Please choose one of: " + ", ".join(keys))
