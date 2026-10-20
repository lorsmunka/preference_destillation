"""Reading and writing the JSON files that hold all the bank's data."""

import json
import os

from config import ACCOUNTS_FILE, DATA_FOLDER, TRANSACTIONS_FILE, USERS_FILE


def load_json(path, default):
    """The data in a JSON file, or `default` when the file does not exist yet."""
    if not os.path.exists(path):
        return default
    with open(path) as file:
        return json.load(file)


def save_json(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_users():
    return load_json(USERS_FILE, {})


def load_accounts():
    return load_json(ACCOUNTS_FILE, {})


def save_accounts(accounts):
    save_json(ACCOUNTS_FILE, accounts)


def load_transactions():
    return load_json(TRANSACTIONS_FILE, [])


def save_transactions(transactions):
    save_json(TRANSACTIONS_FILE, transactions)
