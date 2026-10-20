"""Finding recipes."""

from pantry import missing_ingredients
from recipes import get_recipe
from storage import load_recipes


def recipes_with(ingredient):
    """Names of the recipes that use an ingredient."""
    return sorted(name for name, data in load_recipes().items() if ingredient.lower() in data["ingredients"])


def can_cook_now():
    """Recipes the pantry has everything for, at their normal servings."""
    return sorted(name for name in load_recipes() if not missing_ingredients(get_recipe(name).ingredients))


def quick_recipes(max_steps=4):
    """Recipes with few steps, the shortest first."""
    short = [(len(data["steps"]), name) for name, data in load_recipes().items() if len(data["steps"]) <= max_steps]
    return [name for _, name in sorted(short)]
