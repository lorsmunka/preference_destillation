"""Writing messages: a draft waits in the Drafts folder until it is queued."""

from config import MY_ADDRESS
from messages import Message, new_message_id, now_text
from threads import reply_subject

SIGNATURE = "-- \nSam Taylor\nsam@example.com"


def add_signature(body):
    return body if SIGNATURE in body else body.rstrip() + "\n\n" + SIGNATURE


def new_draft(mailbox, address_book, to_text, subject, body):
    """Save a draft. Returns (draft, problems): unknown names are problems."""
    recipients, problems = address_book.resolve(to_text)
    draft = Message(new_message_id(), MY_ADDRESS, recipients, subject, body)
    mailbox.get_folder("Drafts").add_message(draft)
    return draft, problems


def reply_draft(mailbox, original, body):
    """A signed draft to the sender of `original`, quoting it."""
    quoted = "\n".join("> " + line for line in original.body.split("\n"))
    text = f"{add_signature(body)}\n\n{original.sender} wrote:\n{quoted}"
    draft = Message(new_message_id(), MY_ADDRESS, [original.sender], reply_subject(original.subject), text)
    mailbox.get_folder("Drafts").add_message(draft)
    return draft


def finish_draft(mailbox, draft_id, outbox):
    """Sign a draft and queue it in the outbox. The problems ([] when queued)."""
    drafts = mailbox.get_folder("Drafts")
    draft = drafts.remove_message(draft_id)
    if not draft.recipients or not draft.subject.strip():
        drafts.add_message(draft)
        return ["A message needs a recipient and a subject."]
    draft.body = add_signature(draft.body)
    draft.sent_at = now_text()
    outbox.queue_message(draft)
    return []
