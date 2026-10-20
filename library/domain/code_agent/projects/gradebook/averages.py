"""Averages: of a pupil in a subject, and of the whole class."""

from config import HOMEWORK_WEIGHT, TEST_WEIGHT
from grades import grades_of
from students import students_in_class


def average(scores):
    if not scores:
        return None
    return round(sum(scores) / len(scores), 1)


def subject_average(student_id, subject):
    """Tests count for 70% and homework for 30%; with only one kind, that kind counts fully."""
    tests = average([g["score"] for g in grades_of(student_id, subject) if g["kind"] == "test"])
    homework = average([g["score"] for g in grades_of(student_id, subject) if g["kind"] == "homework"])
    if tests is None:
        return homework
    if homework is None:
        return tests
    return round(tests * TEST_WEIGHT + homework * HOMEWORK_WEIGHT, 1)


def class_average(class_name, subject):
    """The average of the pupils' subject averages (pupils without scores are left out)."""
    results = [subject_average(student.student_id, subject) for student in students_in_class(class_name)]
    return average([result for result in results if result is not None])
