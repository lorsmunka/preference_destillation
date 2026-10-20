"""Who was in class each day."""

from config import MIN_ATTENDANCE
from dates import today
from storage import load_attendance, save_attendance


def mark(student_id, present, day=None):
    """Write down that a pupil was there (True) or not (False) on a day, today by default."""
    attendance = load_attendance()
    attendance.setdefault(day or today(), {})[student_id] = present
    save_attendance(attendance)


def attendance_rate(student_id):
    """The share of school days the pupil was there, from 0 to 1 (None before any day is marked)."""
    days = [marks[student_id] for marks in load_attendance().values() if student_id in marks]
    if not days:
        return None
    return sum(days) / len(days)


def missed_too_much(student_id):
    rate = attendance_rate(student_id)
    return rate is not None and rate < MIN_ATTENDANCE


def absent_today():
    marks = load_attendance().get(today(), {})
    return [student_id for student_id, present in marks.items() if not present]
