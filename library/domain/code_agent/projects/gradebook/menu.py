"""The teacher's menu."""

CHOICES = {
    "1": "Enter a score",
    "2": "Take attendance",
    "3": "Print a report card",
    "4": "Class ranking",
    "5": "Missing homework",
    "6": "Add a pupil",
    "q": "Quit",
}


def show_menu(class_name):
    print(f"\n--- Gradebook, class {class_name} ---")
    for key, label in CHOICES.items():
        print(f" {key}) {label}")


def ask_choice():
    while True:
        choice = input("Choice: ").strip().lower()
        if choice in CHOICES:
            return choice
        print("Unknown choice.")
