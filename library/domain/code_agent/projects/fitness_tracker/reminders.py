"""Reminders: fixed ones on chosen weekdays, and a nudge after a few days without exercise."""

from dataclasses import dataclass
from datetime import date

from config import NUDGE_AFTER_DAYS
from streaks import active_days

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
REMINDER_TIMES = [("monday", 17, "Running club at 18:30."), ("thursday", 6, "Swim before work.")]


@dataclass
class Reminder:
    weekday: str
    hour: int
    message: str

    def is_due(self, now):
        """Due on its weekday, from its hour on."""
        return WEEKDAYS[now.weekday()] == self.weekday and now.hour >= self.hour


def due_reminders(now):
    reminders = [Reminder(*times) for times in REMINDER_TIMES]
    return [reminder.message for reminder in reminders if reminder.is_due(now)]


def inactivity_nudge(log, today):
    """A nudge when the last active day was NUDGE_AFTER_DAYS or more days ago, otherwise None."""
    days = active_days(log)
    if not days:
        return "Log your first workout today!"
    idle_days = (today - date.fromisoformat(max(days))).days
    return f"No exercise for {idle_days} days. A short walk counts too!" if idle_days >= NUDGE_AFTER_DAYS else None


def startup_messages(log, now):
    """What to show when the program starts: the reminders due now and the nudge, if any."""
    return [message for message in due_reminders(now) + [inactivity_nudge(log, now.date())] if message]
