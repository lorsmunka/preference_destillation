"""The mail client in the terminal."""

import getpass

from contacts import AddressBook
from drafts import finish_draft, new_draft, reply_draft
from finder import find_messages
from outbox import Outbox
from rules import apply_rules, load_rules
from spam import report_spam, sweep_inbox
from storage import load_mailbox, save_mailbox


def show(messages):
    for number, message in enumerate(messages, 1):
        print(f"{number:>3} {' ' if message.is_read else '*'} {message.sender:<25} {message.subject}")
    return messages


def main():
    mailbox, address_book, outbox = load_mailbox(), AddressBook(), Outbox()
    print(apply_rules(mailbox, load_rules()), "filed by rules,", sweep_inbox(mailbox, address_book), "to Spam.")
    shown = show(mailbox.get_folder("Inbox").newest_first())
    while (line := input("mail> ").strip()) != "quit":
        command, _, argument = line.partition(" ")
        picked = shown[int(argument) - 1] if argument.isdigit() else None
        if command == "list":
            shown = show(mailbox.get_folder(argument or "Inbox").newest_first())
        elif command == "find":
            shown = show(find_messages(mailbox, argument))
        elif command == "read":
            picked.is_read = True
            print(f"From: {picked.sender}\n{picked.subject}\n\n{picked.body}")
        elif command in ("write", "reply"):
            if command == "write":
                draft, problems = new_draft(mailbox, address_book, input("To: "), input("Subject: "), input("Text: "))
            else:
                draft, problems = reply_draft(mailbox, picked, input("Text: ")), []
            print("\n".join(problems or finish_draft(mailbox, draft.message_id, outbox)) or "Queued.")
        elif command == "spam":
            report_spam(mailbox, picked)
        elif command == "send":
            print(outbox.flush(mailbox, getpass.getpass("Password: ")), "sent.")
    save_mailbox(mailbox)


if __name__ == "__main__":
    main()
