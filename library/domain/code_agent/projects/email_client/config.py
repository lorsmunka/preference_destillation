"""Settings for the mail client."""

DATA_FOLDER = "mail"
MAILBOX_FILE = "mail/mailbox.json"
OUTBOX_FILE = "mail/outbox.json"
CONTACTS_FILE = "mail/contacts.json"
RULES_FILE = "mail/rules.json"
REPUTATION_FILE = "mail/reputation.json"

MY_ADDRESS = "sam@example.com"
SMTP_HOST = "smtp.example.com"
SMTP_PORT = 587

SPAM_THRESHOLD = 5
MAX_RETRIES = 3
RETRY_WAIT_MINUTES = 10
REPLY_PREFIXES = ("re:", "fwd:", "fw:")
