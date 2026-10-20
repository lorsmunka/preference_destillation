"""Bank simulator: log in, then use the menu until you log out."""

from accounts import load_account
from auth import account_of, login
from history import print_statement
from menu import ask_choice, show_menu
from reports import print_month_summary
from transfers import transfer
from utils import ask_number, format_money


def handle_choice(choice, account):
    """Do what the user picked in the menu. Returns False when the user logs out."""
    if choice == "1":
        print("Balance: " + format_money(account.get_balance()))
    elif choice == "2":
        problem = account.deposit(ask_number("How much to deposit? "))
        print(problem or "Done.")
    elif choice == "3":
        problem = account.withdraw(ask_number("How much to withdraw? "))
        print(problem or "Done.")
    elif choice == "4":
        to_number = input("Send to account number: ").strip()
        problem = transfer(account.number, to_number, ask_number("How much? "))
        print(problem or "Sent.")
    elif choice == "5":
        print_statement(account.number)
    elif choice == "6":
        print_month_summary(account.number)
    elif choice == "0":
        print("Goodbye!")
        return False
    return True


def main():
    username = login()
    if username is None:
        return
    keep_going = True
    while keep_going:
        account = load_account(account_of(username))   # fresh from the file: a transfer may have changed it
        show_menu()
        keep_going = handle_choice(ask_choice(), account)


if __name__ == "__main__":
    main()
