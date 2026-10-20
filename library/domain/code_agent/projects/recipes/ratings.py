"""The family rates each meal from 1 to 5."""

from config import RATINGS_FILE
from storage import keep_json, open_json


def rate(recipe_name, person, score):
    """Save one person's score for a recipe (a new score replaces their old one)."""
    if not 1 <= score <= 5:
        return False
    ratings = open_json(RATINGS_FILE, {})
    ratings.setdefault(recipe_name, {})[person] = score
    keep_json(RATINGS_FILE, ratings)
    return True


def family_favourite():
    """The recipe with the best average score (at least two people must have rated it)."""
    best, best_average = None, 0
    for recipe_name, scores in open_json(RATINGS_FILE, {}).items():
        if len(scores) >= 2:
            average = sum(scores.values()) / len(scores)
            if average > best_average:
                best, best_average = recipe_name, average
    return best
