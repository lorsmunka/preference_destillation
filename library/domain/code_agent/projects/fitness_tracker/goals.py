"""Weekly goals (minutes of exercise, number of workouts, distance) and how far along each one is."""

from dataclasses import dataclass

from config import WEEKLY_GOALS


@dataclass
class WeeklyGoal:
    kind: str        # "minutes", "workouts" or "km"
    target: float

    def amount(self, workouts):
        """How much of this goal the week's workouts add up to."""
        if self.kind == "minutes":
            return sum(workout.minutes for workout in workouts)
        if self.kind == "km":
            return sum(workout.km for workout in workouts)
        return len(workouts)

    def progress(self, workouts):
        """The share of the target reached, from 0 to 1 (going past the target still counts as 1)."""
        return min(1.0, self.amount(workouts) / self.target)


def weekly_goals():
    return [WeeklyGoal(kind, target) for kind, target in WEEKLY_GOALS.items()]


def progress_bar(fraction, width=20):
    """0.4 -> "[########            ] 40%"."""
    filled = round(fraction * width)
    return "[" + "#" * filled + " " * (width - filled) + f"] {fraction:.0%}"
