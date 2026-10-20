"""Scores of tests and homework, and the letter each score gets."""

from config import LETTER_BOUNDARIES
from dates import today
from storage import load_grades, save_grades
from validation import check_score


def record_grade(student_id, subject, score, kind="test"):
    """Save one score. `kind` is "test" or "homework". None when it worked."""
    problem = check_score(score)
    if problem:
        return problem
    grades = load_grades()
    grades.append({"student": student_id, "subject": subject, "score": score, "kind": kind, "date": today()})
    save_grades(grades)
    return None


def grades_of(student_id, subject=None):
    """The scores of one pupil, in one subject or in all of them."""
    return [grade for grade in load_grades()
            if grade["student"] == student_id and (subject is None or grade["subject"] == subject)]


def letter_for(score):
    """The letter grade of a score: A for 90 and up, down to F."""
    for lowest, letter in LETTER_BOUNDARIES:
        if score >= lowest:
            return letter
    return "F"


def latest_grade(student_id, subject):
    found = grades_of(student_id, subject)
    return found[-1] if found else None
