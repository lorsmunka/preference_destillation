"""Searching the folders by words and by sender ("from:anna")."""

SKIPPED_FOLDERS = ("Spam", "Trash")


def matches_query(message, query):
    """True when the message has every word of the query (a from: word in its sender)."""
    text = (message.subject + " " + message.body).lower()
    for word in query.lower().split():
        if word.startswith("from:"):
            if word[5:] not in message.sender.lower():
                return False
        elif word not in text:
            return False
    return True


def find_messages(mailbox, query):
    """The matches in every folder but Spam and Trash, newest first."""
    found = [message for name, folder in mailbox.folders.items() if name not in SKIPPED_FOLDERS
             for message in folder.messages if matches_query(message, query)]
    return sorted(found, key=lambda message: message.sent_time, reverse=True)
