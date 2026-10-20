"""Logging in: the username and PIN are looked up in users.json."""

from config import MAX_LOGIN_TRIES
from storage import load_users


def check_login(username, pin):
    """True when the user exists and the PIN matches."""
    users = load_users()
    if username not in users:
        return False
    return users[username]["pin"] == pin


def login():
    """Ask for a username and PIN until they are right. Returns the username, or None after too many tries."""
    for attempt in range(MAX_LOGIN_TRIES):
        username = input("Username: ").strip()
        pin = input("PIN: ").strip()
        if check_login(username, pin):
            print(f"Welcome, {username}!")
            return username
        tries_left = MAX_LOGIN_TRIES - attempt - 1
        print(f"Wrong username or PIN. {tries_left} tries left.")
    print("Too many wrong tries. Goodbye.")
    return None


def account_of(username):
    """The account number that belongs to this user."""
    return load_users()[username]["account"]
