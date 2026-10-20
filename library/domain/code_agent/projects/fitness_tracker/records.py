"""Personal records: the longest, farthest and fastest workout of each exercise, and the best step day."""

from units import format_distance, format_pace


def exercise_records(log, exercise):
    """The longest, farthest and fastest workout of one exercise (None while there is none)."""
    workouts = [workout for workout in log.workouts if workout.exercise == exercise]
    with_distance = [workout for workout in workouts if workout.km]
    return {"longest": max(workouts, key=lambda workout: workout.minutes, default=None),
            "farthest": max(with_distance, key=lambda workout: workout.km, default=None),
            "fastest": min(with_distance, key=lambda workout: workout.pace, default=None)}


def new_records(log, workout):
    """The records a just-added workout set. A tie keeps the older record: max and min keep the first."""
    return [name for name, best in exercise_records(log, workout.exercise).items() if best is workout]


def records_text(log, unit):
    """One line per exercise with its records, then the day with the most steps."""
    lines = []
    for exercise in sorted({workout.exercise for workout in log.workouts}):
        best = exercise_records(log, exercise)
        line = f"{exercise}: longest {best['longest'].minutes} min"
        if best["farthest"]:
            line += f", farthest {format_distance(best['farthest'].km, unit)}"
            line += f", fastest {format_pace(best['fastest'].pace, unit)}"
        lines.append(line)
    if log.steps:
        day = max(log.steps, key=log.steps.get)
        lines.append(f"most steps: {log.steps[day]} on {day}")
    return lines
