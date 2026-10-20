"""The outbox: messages wait here until the mail server takes them."""

from datetime import datetime, timedelta

from config import MAX_RETRIES, OUTBOX_FILE, RETRY_WAIT_MINUTES
from messages import DATE_FORMAT, message_from_dict, now_text
from storage import read_json, write_json
from transport import send_now


class Outbox:
    def __init__(self):
        self.entries = read_json(OUTBOX_FILE, [])

    def queue_message(self, message):
        self.entries.append({"message": message.to_dict(), "tries": 0, "next_try": now_text()})
        self.save()

    def retry_later(self, entry, now):
        """Count a failed try; the wait doubles each time. False after MAX_RETRIES."""
        entry["tries"] += 1
        wait = RETRY_WAIT_MINUTES * 2 ** (entry["tries"] - 1)
        entry["next_try"] = (now + timedelta(minutes=wait)).strftime(DATE_FORMAT)
        return entry["tries"] < MAX_RETRIES

    def flush(self, mailbox, password):
        """Send what is due: sent mail goes to Sent, hopeless mail back to Drafts."""
        now, sent = datetime.now(), 0
        for entry in list(self.entries):
            if datetime.strptime(entry["next_try"], DATE_FORMAT) > now:
                continue
            message = message_from_dict(entry["message"])
            if send_now(message, password) is None:
                mailbox.get_folder("Sent").add_message(message)
                self.entries.remove(entry)
                sent += 1
            elif not self.retry_later(entry, now):
                mailbox.get_folder("Drafts").add_message(message)
                self.entries.remove(entry)
        self.save()
        return sent

    def save(self):
        write_json(OUTBOX_FILE, self.entries)
