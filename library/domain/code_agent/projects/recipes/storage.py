"""Recipes, the pantry and the week's plan are JSON files."""

import json
import os

from config import DATA_FOLDER, PANTRY_FILE, PLAN_FILE, RECIPES_FILE


def open_json(path, empty):
    if not os.path.exists(path):
        return empty
    with open(path) as file:
        return json.load(file)


def keep_json(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_recipes():
    return open_json(RECIPES_FILE, {})


def save_recipes(recipes):
    keep_json(RECIPES_FILE, recipes)


def load_pantry():
    return open_json(PANTRY_FILE, {})


def save_pantry(pantry):
    keep_json(PANTRY_FILE, pantry)


def load_plan():
    return open_json(PLAN_FILE, {})


def save_plan(plan):
    keep_json(PLAN_FILE, plan)
