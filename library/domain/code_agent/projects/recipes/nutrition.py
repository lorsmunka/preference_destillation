"""Calories: per serving, per day and for the week."""

from config import DAILY_CALORIES, DAYS
from meal_plan import meal_for
from recipes import get_recipe


def meal_calories(day):
    """Calories of one serving of that day's meal (0 when nothing is planned)."""
    meal = meal_for(day)
    if meal is None:
        return 0
    return get_recipe(meal["recipe"]).calories_per_serving()


def share_of_day(calories):
    """How much of a day's calories a meal is, from 0 to 1."""
    return calories / DAILY_CALORIES


def week_report():
    """Each day's dinner calories and its share of the day."""
    return [(day, meal_calories(day), round(share_of_day(meal_calories(day)), 2)) for day in DAYS]


def heaviest_day():
    report = week_report()
    return max(report, key=lambda row: row[1])[0] if report else None
