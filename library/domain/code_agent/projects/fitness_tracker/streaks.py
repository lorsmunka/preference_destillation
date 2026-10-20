"""Streaks: active days in a row, a day being active with a workout or with the step goal met."""

from datetime import date, timedelta

from steps import goal_met


def active_days(log):
    days = {workout.day for workout in log.workouts}
    return days | {day for day, count in log.steps.items() if goal_met(count)}


def current_streak(log, today):
    """Active days in a row up to today. Today not being active yet does not break the streak."""
    days, streak = active_days(log), 0
    day = today if today.isoformat() in days else today - timedelta(days=1)
    while day.isoformat() in days:
        streak += 1
        day -= timedelta(days=1)
    return streak


def longest_streak(log):
    """The most active days in a row there have ever been."""
    longest = run = 0
    previous = None
    for day in sorted(date.fromisoformat(text) for text in active_days(log)):
        run = run + 1 if previous and day - previous == timedelta(days=1) else 1
        longest, previous = max(longest, run), day
    return longest
