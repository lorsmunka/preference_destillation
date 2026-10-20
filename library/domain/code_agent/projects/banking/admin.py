"""Tools for the bank staff: opening accounts and the end-of-month run."""

from accounts import Account, load_account
from config import USERS_FILE
from fees import charge_monthly_fee
from interest import add_interest
from storage import load_accounts, load_users, save_json
from utils import new_id


def open_account(username, pin, kind="checking"):
    """Create a user with a new account that starts at zero."""
    users = load_users()
    if username in users:
        return None
    number = new_id("A")
    users[username] = {"pin": pin, "account": number}
    save_json(USERS_FILE, users)
    Account(number, username, 0, kind).save()
    return number


def end_of_month():
    """Charge the monthly fees and add interest on every account."""
    for number in load_accounts():
        account = load_account(number)
        charge_monthly_fee(account)
        add_interest(account)


def total_money_in_bank():
    return sum(data["balance"] for data in load_accounts().values())
