"""One email message, as the client keeps it in its folders."""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

DATE_FORMAT = "%Y-%m-%d %H:%M"


@dataclass
class Message:
    message_id: str
    sender: str
    recipients: list
    subject: str
    body: str
    sent_at: str = ""   # empty while it is a draft
    is_read: bool = False

    @property
    def sent_time(self):
        return datetime.strptime(self.sent_at, DATE_FORMAT) if self.sent_at else datetime.min

    def to_dict(self):
        return asdict(self)


def message_from_dict(data):
    return Message(**data)


def now_text():
    return datetime.now().strftime(DATE_FORMAT)


def new_message_id():
    return f"<{uuid.uuid4().hex}@example.com>"
