"""The weekly report: time, distance and calories per exercise, steps, goals, streak and weight trend."""

from dates import week_days
from goals import progress_bar, weekly_goals
from steps import steps_to_km, week_steps
from streaks import current_streak
from units import format_distance
from weight import weight_trend


def exercise_totals(workouts):
    """exercise -> (minutes, km, calories), added up over the workouts."""
    totals = {}
    for workout in workouts:
        minutes, km, calories = totals.get(workout.exercise, (0, 0.0, 0.0))
        totals[workout.exercise] = (minutes + workout.minutes, km + workout.km, calories + workout.calories)
    return totals


def weekly_report(log, start, today, unit):
    """The lines of the report for the week that starts on `start`."""
    days = week_days(start)
    workouts = log.workouts_on(days)
    lines = [f"Week of {days[0]}"]
    for exercise, (minutes, km, calories) in sorted(exercise_totals(workouts).items()):
        distance = f" {format_distance(km, unit)}" if km else ""
        lines.append(f"  {exercise:<9} {minutes} min{distance}, {calories:.0f} kcal")
    steps = week_steps(log, days)
    lines.append(f"  steps     {steps['total']} ({format_distance(steps_to_km(steps['total']), unit)}), "
                 f"goal met on {steps['goal_days']} days, best day {steps['best_day']}")
    for goal in weekly_goals():
        lines.append(f"  {goal.kind:<9} {progress_bar(goal.progress(workouts))}")
    lines.append(f"  streak    {current_streak(log, today)} days, weight {weight_trend(log)}")
    return lines
