"""Homework handed in or missing."""

from grades import grades_of
from students import students_in_class


def handed_in(student_id, subject):
    """How many homework scores the pupil has in a subject."""
    return len([grade for grade in grades_of(student_id, subject) if grade["kind"] == "homework"])


def missing_homework(class_name, subject, expected):
    """Pupils who handed in fewer than `expected` homeworks, with how many are missing."""
    missing = []
    for student in students_in_class(class_name):
        count = handed_in(student.student_id, subject)
        if count < expected:
            missing.append((student.full_name(), expected - count))
    return missing


def print_missing(class_name, subject, expected):
    for name, count in missing_homework(class_name, subject, expected):
        print(f"{name} is missing {count} homework{'s' if count > 1 else ''} in {subject}")
