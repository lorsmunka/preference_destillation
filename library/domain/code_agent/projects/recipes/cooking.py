"""Cooking a planned meal: show the steps and take the ingredients out of the pantry."""

from meal_plan import meal_for
from pantry import missing_ingredients, use_ingredient
from recipes import get_recipe


def cook(day):
    """Cook the day's meal. None when it worked, otherwise why not."""
    meal = meal_for(day)
    if meal is None:
        return "Nothing is planned for that day."
    recipe = get_recipe(meal["recipe"])
    needed = recipe.scaled(meal["servings"])
    missing = missing_ingredients(needed)
    if missing:
        return "Missing: " + ", ".join(sorted(missing))
    for ingredient, (amount, unit) in needed.items():
        use_ingredient(ingredient, amount, unit)
    show_steps(recipe)
    return None


def show_steps(recipe):
    print(f"--- {recipe.name} ---")
    for number, step in enumerate(recipe.steps, 1):
        print(f"{number}. {step}")
