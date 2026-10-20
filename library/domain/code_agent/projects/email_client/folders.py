"""Folders hold messages; the Mailbox holds the folders."""

STANDARD_FOLDERS = ("Inbox", "Sent", "Drafts", "Spam", "Trash")


class Folder:
    def __init__(self, name):
        self.name = name
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def remove_message(self, message_id):
        """Take a message out and return it (None when it is not here)."""
        for message in self.messages:
            if message.message_id == message_id:
                self.messages.remove(message)
                return message
        return None

    def newest_first(self):
        return sorted(self.messages, key=lambda message: message.sent_time, reverse=True)


class Mailbox:
    def __init__(self):
        self.folders = {name: Folder(name) for name in STANDARD_FOLDERS}

    def get_folder(self, name):
        if name not in self.folders:
            self.folders[name] = Folder(name)
        return self.folders[name]

    def move(self, message_id, from_name, to_name):
        message = self.get_folder(from_name).remove_message(message_id)
        if message is not None:
            self.get_folder(to_name).add_message(message)
        return message
