"""Handing a message to the mail server over SMTP."""

import smtplib
from email.message import EmailMessage

from config import MY_ADDRESS, SMTP_HOST, SMTP_PORT

TIMEOUT_SECONDS = 20


def to_mime(message):
    mime = EmailMessage()
    mime["From"] = message.sender
    mime["To"] = ", ".join(message.recipients)
    mime["Subject"] = message.subject
    mime.set_content(message.body)
    return mime


def send_now(message, password):
    """None when the server took the message, otherwise the error text."""
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(MY_ADDRESS, password)
            server.send_message(to_mime(message))
    except (smtplib.SMTPException, OSError) as error:
        return str(error)
    return None
