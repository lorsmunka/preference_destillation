"""Checks on what the teacher types in."""

from config import MAX_SCORE
from storage import load_students


def check_score(score):
    """None for a whole score from 0 to MAX_SCORE, otherwise what is wrong."""
    if not isinstance(score, int):
        return "Scores are whole numbers."
    if score < 0 or score > MAX_SCORE:
        return f"A score is between 0 and {MAX_SCORE}."
    return None


def student_exists(student_id):
    return student_id in load_students()


def read_score(text):
    """The number in what was typed, or None ("87", " 87 " and "87%" all give 87)."""
    text = text.strip().rstrip("%")
    return int(text) if text.isdigit() else None
