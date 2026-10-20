"""Counting requests: per hour, path, IP address and status class."""

from collections import Counter, defaultdict


def group_by_hour(entries):
    """{"2026-10-21 13:00": [entries], ...}, earliest hour first."""
    groups = defaultdict(list)
    for entry in entries:
        groups[entry.hour_label].append(entry)
    return dict(sorted(groups.items()))


def top_paths(entries, count):
    """[(path, requests), ...], the most requested first."""
    return Counter(entry.path for entry in entries).most_common(count)


def top_ips(entries, count):
    """The IP addresses that sent the most requests."""
    return Counter(entry.ip for entry in entries).most_common(count)


def status_counts(entries):
    """{"2xx": 950, "4xx": 40, "5xx": 10}."""
    return dict(sorted(Counter(entry.status_class for entry in entries).items()))
