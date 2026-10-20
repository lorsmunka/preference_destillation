"""The mailbox and the client's other data live in JSON files."""

import json
import os

from config import DATA_FOLDER, MAILBOX_FILE
from folders import Mailbox
from messages import message_from_dict


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_mailbox():
    """The file holds {"Inbox": [message, ...], "Sent": [...], ...}."""
    mailbox = Mailbox()
    for folder_name, records in read_json(MAILBOX_FILE, {}).items():
        for record in records:
            mailbox.get_folder(folder_name).add_message(message_from_dict(record))
    return mailbox


def save_mailbox(mailbox):
    write_json(MAILBOX_FILE, {name: [message.to_dict() for message in folder.messages]
                              for name, folder in mailbox.folders.items()})
