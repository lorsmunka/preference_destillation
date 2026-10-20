"""Days are kept as "YYYY-MM-DD" text; a week starts on Monday."""

from datetime import date, timedelta


def parse_day(text, today):
    """"today" (or nothing), "yesterday" or "YYYY-MM-DD" -> a date."""
    text = text.strip().lower()
    if text in ("", "today"):
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    return date.fromisoformat(text)


def week_start(day):
    """The Monday of the week a day is in."""
    return day - timedelta(days=day.weekday())


def week_days(start):
    """The seven days of the week that starts on `start`, as text."""
    return [(start + timedelta(days=offset)).isoformat() for offset in range(7)]
