"""The plan: which recipe is cooked on which day of the week."""

from config import DAYS, MAX_SAME_RECIPE_PER_WEEK
from recipes import get_recipe
from storage import load_plan, save_plan


def plan_meal(day, recipe_name, servings):
    """Put a recipe on a day. None when it worked, otherwise why not."""
    if day not in DAYS:
        return f"{day} is not a day of the week."
    if get_recipe(recipe_name) is None:
        return f"There is no recipe called {recipe_name}."
    plan = load_plan()
    times = sum(1 for meal in plan.values() if meal["recipe"] == recipe_name)
    if times >= MAX_SAME_RECIPE_PER_WEEK and plan.get(day, {}).get("recipe") != recipe_name:
        return "That would be too often in one week."
    plan[day] = {"recipe": recipe_name, "servings": servings}
    save_plan(plan)
    return None


def meal_for(day):
    return load_plan().get(day)


def clear_week():
    save_plan({})


def print_week():
    plan = load_plan()
    for day in DAYS:
        meal = plan.get(day)
        print(f"{day:<10} " + (f"{meal['recipe']} for {meal['servings']}" if meal else "-"))
