"""Reading the log: a regular expression turns a line into a LogEntry."""

import re
from datetime import datetime

from entries import LogEntry

LINE_PATTERN = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^ \]]+)[^\]]*\] '
    r'"(?P<method>[A-Z]+) (?P<path>\S+)[^"]*" '
    r'(?P<status>\d{3}) (?P<size>\d+|-) (?P<seconds>\d+\.\d+)')
TIME_FORMAT = "%d/%b/%Y:%H:%M:%S"   # the time zone is ignored


def parse_line(line):
    """A LogEntry, or None when the line is not an access log line."""
    found = LINE_PATTERN.match(line)
    if found is None:
        return None
    time = datetime.strptime(found["time"], TIME_FORMAT)
    size = 0 if found["size"] == "-" else int(found["size"])
    return LogEntry(found["ip"], time, found["method"], found["path"],
                    int(found["status"]), size, float(found["seconds"]))


def read_entries(path):
    """The entries of a log file, and the number of unreadable lines."""
    entries, bad_lines = [], 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            entry = parse_line(line.strip())
            if entry is not None:
                entries.append(entry)
            elif line.strip():
                bad_lines += 1
    return entries, bad_lines
