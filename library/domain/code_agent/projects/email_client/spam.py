"""Spam scoring: spam words in the message plus the sender's reputation."""

import re

from config import REPUTATION_FILE, SPAM_THRESHOLD
from storage import read_json, write_json

SPAM_WORDS = {"winner": 3, "lottery": 3, "bitcoin": 2, "urgent": 1, "free": 1}
CONTACT_POINTS = -4
POINTS_PER_REPORT = 2


def word_points(text):
    """Points for the spam words in the text; each counts once, as a whole word."""
    text = text.lower()
    return sum(points for word, points in SPAM_WORDS.items() if re.search(rf"\b{word}\b", text))


def sender_points(sender, address_book):
    """Contacts are trusted; other senders get points for each spam report."""
    if address_book.knows(sender):
        return CONTACT_POINTS
    return POINTS_PER_REPORT * read_json(REPUTATION_FILE, {}).get(sender.lower(), 0)


def spam_score(message, address_book):
    words = word_points(message.subject + " " + message.body)
    return words + sender_points(message.sender, address_book)


def sweep_inbox(mailbox, address_book):
    """Move every inbox message that scores SPAM_THRESHOLD or more to Spam."""
    moved = 0
    for message in list(mailbox.get_folder("Inbox").messages):
        if spam_score(message, address_book) >= SPAM_THRESHOLD:
            mailbox.move(message.message_id, "Inbox", "Spam")
            moved += 1
    return moved


def report_spam(mailbox, message):
    """The user's verdict: to Spam with it, and one more report for its sender."""
    reports = read_json(REPUTATION_FILE, {})
    sender = message.sender.lower()
    reports[sender] = reports.get(sender, 0) + 1
    write_json(REPUTATION_FILE, reports)
    mailbox.move(message.message_id, "Inbox", "Spam")
