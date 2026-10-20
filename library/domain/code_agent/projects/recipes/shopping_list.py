"""The shopping list for the week: everything the plan needs that the pantry does not have."""

from pantry import missing_ingredients
from recipes import get_recipe
from storage import load_plan
from units import nice_amount, to_base


def week_needs():
    """All ingredients of the planned meals, scaled to their servings and added up per ingredient."""
    needs = {}
    for meal in load_plan().values():
        recipe = get_recipe(meal["recipe"])
        for ingredient, (amount, unit) in recipe.scaled(meal["servings"]).items():
            base_amount, base_unit = to_base(amount, unit)
            have, _ = needs.get(ingredient, (0, base_unit))
            needs[ingredient] = (have + base_amount, base_unit)
    return needs


def build_shopping_list():
    """What to buy: the week's needs minus the pantry, sorted by name."""
    return sorted(missing_ingredients(week_needs()).items())


def print_shopping_list():
    items = build_shopping_list()
    if not items:
        print("You have everything!")
    for ingredient, (amount, unit) in items:
        print(f"[ ] {ingredient}: {nice_amount(amount, unit)}")
