"""Filter rules: move incoming messages to a folder by their sender or subject."""

from config import RULES_FILE
from storage import read_json


def load_rules():
    """Rules like {"field": "sender", "pattern": "shop", "folder": "Offers"}."""
    return read_json(RULES_FILE, [])


def rule_folder(message, rules):
    """The folder of the first rule that matches the message, or None."""
    for rule in rules:
        value = message.sender if rule["field"] == "sender" else message.subject
        if rule["pattern"].lower() in value.lower():
            return rule["folder"]
    return None


def apply_rules(mailbox, rules):
    """Move each inbox message to the folder its first matching rule names."""
    moved = 0
    for message in list(mailbox.get_folder("Inbox").messages):
        folder_name = rule_folder(message, rules)
        if folder_name is not None:
            mailbox.move(message.message_id, "Inbox", folder_name)
            moved += 1
    return moved
