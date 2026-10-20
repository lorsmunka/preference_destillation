"""The menu a member sees after scanning their card."""

OPTIONS = [
    ("1", "Search by title"),
    ("2", "Search by author"),
    ("3", "Borrow a book"),
    ("4", "Return a book"),
    ("5", "Reserve a book"),
    ("6", "My loans"),
    ("0", "Done"),
]


def show_menu(name):
    print(f"\nHello {name}, what would you like to do?")
    for key, label in OPTIONS:
        print(f"  {key}. {label}")


def ask_choice():
    keys = [key for key, _ in OPTIONS]
    while True:
        choice = input("> ").strip()
        if choice in keys:
            return choice
        print("Please type one of " + " ".join(keys))
