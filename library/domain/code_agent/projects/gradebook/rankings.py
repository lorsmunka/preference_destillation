"""Who is top of the class."""

from averages import average, subject_average
from students import students_in_class
from subjects import SUBJECTS


def overall_average(student_id):
    """The average over every subject the pupil has scores in."""
    results = [subject_average(student_id, subject) for subject in SUBJECTS]
    return average([result for result in results if result is not None])


def rank_students(class_name):
    """The pupils of a class with their overall average, best first; pupils without scores last."""
    ranked = [(student, overall_average(student.student_id)) for student in students_in_class(class_name)]
    return sorted(ranked, key=lambda pair: -1 if pair[1] is None else pair[1], reverse=True)


def top_student(class_name):
    ranked = rank_students(class_name)
    if not ranked or ranked[0][1] is None:
        return None
    return ranked[0][0]


def print_ranking(class_name):
    for place, (student, result) in enumerate(rank_students(class_name), 1):
        print(f"{place:>2}. {student.full_name():<25} {result if result is not None else '-'}")
