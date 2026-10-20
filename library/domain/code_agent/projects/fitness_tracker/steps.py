"""Daily steps: the step goal, the distance walked and the step numbers of a week."""

from config import DAILY_STEP_GOAL, STRIDE_KM


def steps_to_km(steps):
    return steps * STRIDE_KM


def goal_met(steps):
    return steps >= DAILY_STEP_GOAL


def week_steps(log, days):
    """The total, the best day and on how many days the goal was met, for the given days."""
    counts = {day: log.steps.get(day, 0) for day in days}
    best_day = max(counts, key=counts.get) if any(counts.values()) else "none"
    return {"total": sum(counts.values()), "best_day": best_day,
            "goal_days": sum(goal_met(count) for count in counts.values())}
