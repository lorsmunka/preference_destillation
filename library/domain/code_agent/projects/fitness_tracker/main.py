"""Fitness tracker in the terminal: log workouts, steps and weight; see records and the weekly report."""

from datetime import date, datetime

from activity_log import load_log
from config import UNIT
from dates import parse_day, week_start
from records import new_records, records_text
from reminders import startup_messages
from report import weekly_report
from units import parse_distance
from workouts import CALORIES_PER_MINUTE

MENU = "w=workout  s=steps  b=body weight  r=report  p=records  u=km/miles  q=quit"


def log_workout(log, today):
    """Ask for a workout, save it and say which personal records it set."""
    exercise = input(f"Exercise ({', '.join(CALORIES_PER_MINUTE)}): ").strip().lower()
    if exercise not in CALORIES_PER_MINUTE:
        print("Unknown exercise.")
        return
    day = parse_day(input("Day (today, yesterday or YYYY-MM-DD): "), today).isoformat()
    minutes = int(input("Minutes: "))
    km = parse_distance(input("Distance, like 5 km or 3 mi (Enter for none): ") or "0")
    workout = log.add_workout(day, exercise, minutes, km)
    records = new_records(log, workout)
    print(f"Saved: {workout.calories:.0f} kcal." + (f" New record: {', '.join(records)}!" if records else ""))


def main():
    log, today, unit = load_log(), date.today(), UNIT
    for message in startup_messages(log, datetime.now()):
        print(message)
    while True:
        choice = input(f"\n{MENU}\n> ").strip().lower()
        if choice == "q":
            break
        if choice == "w":
            log_workout(log, today)
        elif choice == "s":
            log.add_steps(parse_day(input("Day: "), today).isoformat(), int(input("Steps: ")))
        elif choice == "b":
            log.record_weight(parse_day(input("Day: "), today).isoformat(), float(input("Weight in kg: ")))
        elif choice == "r":
            print("\n".join(weekly_report(log, week_start(today), today, unit)))
        elif choice == "p":
            print("\n".join(records_text(log, unit)) or "No workouts yet.")
        elif choice == "u":
            unit = "mi" if unit == "km" else "km"
        log.save()


if __name__ == "__main__":
    main()
