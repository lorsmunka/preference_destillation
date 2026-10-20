"""Threads: messages whose subjects match once "Re:" and "Fwd:" are off."""

from config import REPLY_PREFIXES


def base_subject(subject):
    """Turn "Re: RE: Fwd: Lunch?" into "lunch?"."""
    subject = subject.strip()
    while subject.lower().startswith(REPLY_PREFIXES):
        subject = subject.split(":", 1)[1].strip()
    return subject.lower()


def reply_subject(subject):
    """Put "Re: " in front of the subject, but never "Re: Re:"."""
    return subject if subject.lower().startswith("re:") else "Re: " + subject


def group_threads(messages):
    """Base subject -> the messages of that thread, oldest first."""
    threads = {}
    for message in messages:
        threads.setdefault(base_subject(message.subject), []).append(message)
    for thread in threads.values():
        thread.sort(key=lambda message: message.sent_time)
    return threads


def thread_list(messages):
    """(subject, count) per thread, the newest thread first."""
    threads = sorted(group_threads(messages).values(), key=lambda thread: thread[-1].sent_time, reverse=True)
    return [(thread[0].subject, len(thread)) for thread in threads]
