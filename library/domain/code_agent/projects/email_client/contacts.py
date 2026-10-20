"""The address book: names for email addresses."""

import re

from config import CONTACTS_FILE
from storage import read_json

ADDRESS_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


def is_valid_address(address):
    return ADDRESS_PATTERN.fullmatch(address) is not None


class AddressBook:
    def __init__(self):
        self.people = read_json(CONTACTS_FILE, {})   # name -> address

    def knows(self, address):
        return address.lower() in self.people.values()

    def resolve(self, text):
        """Turn "anna, bob@mail.com" into (addresses, problems)."""
        addresses, problems = [], []
        for entry in text.lower().split(","):
            entry = entry.strip()
            if entry in self.people:
                addresses.append(self.people[entry])
            elif is_valid_address(entry):
                addresses.append(entry)
            elif entry:
                problems.append(f"Unknown recipient: {entry}")
        return addresses, problems
