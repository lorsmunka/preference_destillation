"""Star ratings customers give products."""

from config import REVIEWS_FILE
from storage import read_json, write_json


def add_review(product_id, email, stars, text=""):
    """Save a review of 1 to 5 stars; a customer's new review replaces their old one."""
    if stars < 1 or stars > 5:
        return False
    reviews = read_json(REVIEWS_FILE, {})
    reviews.setdefault(product_id, {})[email] = {"stars": stars, "text": text}
    write_json(REVIEWS_FILE, reviews)
    return True


def average_rating(product_id):
    """The average number of stars, to one decimal (None without reviews)."""
    ratings = [review["stars"] for review in read_json(REVIEWS_FILE, {}).get(product_id, {}).values()]
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 1)


def stars_text(rating):
    """3.6 -> "****" (rounded to whole stars)."""
    return "*" * round(rating) if rating else "no reviews"
