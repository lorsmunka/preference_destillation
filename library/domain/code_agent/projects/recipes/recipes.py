"""A Recipe: its ingredients for a number of servings, and the steps."""

from config import DEFAULT_SERVINGS
from storage import load_recipes, save_recipes


class Recipe:
    def __init__(self, name, ingredients, servings=DEFAULT_SERVINGS, steps=None, calories=0):
        self.name = name
        self.ingredients = ingredients   # ingredient -> (amount, unit), e.g. "flour": (200, "g")
        self.servings = servings
        self.steps = steps or []
        self.calories = calories         # for the whole recipe

    def scaled(self, servings):
        """The ingredients for a different number of servings."""
        factor = servings / self.servings
        return {name: (round(amount * factor, 1), unit) for name, (amount, unit) in self.ingredients.items()}

    def calories_per_serving(self):
        return round(self.calories / self.servings)

    def save(self):
        recipes = load_recipes()
        recipes[self.name] = {"ingredients": self.ingredients, "servings": self.servings,
                              "steps": self.steps, "calories": self.calories}
        save_recipes(recipes)


def get_recipe(name):
    data = load_recipes().get(name)
    if data is None:
        return None
    ingredients = {item: tuple(value) for item, value in data["ingredients"].items()}
    return Recipe(name, ingredients, data["servings"], data["steps"], data["calories"])
