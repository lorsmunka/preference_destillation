"""Which entries to look at: a time range, a status, a path."""

from dataclasses import dataclass
from datetime import datetime

DATE_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d")


@dataclass
class EntryFilter:
    since: datetime = None
    until: datetime = None   # the first time left out
    status: str = ""         # "404", or a class like "5xx"
    path_prefix: str = ""

    def matches(self, entry):
        """True when the entry passes every filter that is set."""
        if self.since and entry.time < self.since:
            return False
        if self.until and entry.time >= self.until:
            return False
        if self.status and self.status not in (str(entry.status), entry.status_class):
            return False
        return entry.path.startswith(self.path_prefix)

    def apply(self, entries):
        return [entry for entry in entries if self.matches(entry)]


def parse_date(text):
    """"2026-10-21" or "2026-10-21 13:30" -> a datetime."""
    if not text:
        return None
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            pass
    raise ValueError(f"not a date: {text}")


def filter_from_args(args):
    since, until = parse_date(args.since), parse_date(args.until)
    return EntryFilter(since, until, args.status or "", args.path_prefix)
