"""The report card printed at the end of a term."""

from attendance import attendance_rate, missed_too_much
from averages import subject_average
from config import PASS_MARK
from grades import letter_for
from subjects import SUBJECTS


def build_report_card(student):
    """One line per subject with the average and the letter, plus attendance."""
    lines = []
    for subject in SUBJECTS:
        result = subject_average(student.student_id, subject)
        if result is None:
            lines.append((subject, "-", "-", ""))
            continue
        note = "" if result >= PASS_MARK else "needs help"
        lines.append((subject, result, letter_for(result), note))
    return {"name": student.full_name(), "lines": lines, "attendance": attendance_rate(student.student_id),
            "warning": missed_too_much(student.student_id)}


def print_report_card(student):
    card = build_report_card(student)
    print(f"Report card: {card['name']}")
    for subject, result, letter, note in card["lines"]:
        print(f"  {subject:<12} {result:>5} {letter:>2}  {note}")
    if card["attendance"] is not None:
        print(f"  attendance {card['attendance']:.0%}" + ("  (too many days missed)" if card["warning"] else ""))
