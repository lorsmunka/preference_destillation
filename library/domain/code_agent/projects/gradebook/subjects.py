"""The subjects taught, and who teaches them."""

SUBJECTS = ["Maths", "English", "Science", "History", "Art"]

TEACHER_OF = {
    "Maths": "Mr Novak",
    "English": "Ms Reid",
    "Science": "Dr Okafor",
    "History": "Ms Reid",
    "Art": "Mr Lind",
}


def subject_teacher(subject):
    return TEACHER_OF.get(subject, "unknown")


def subjects_taught_by(teacher):
    """Every subject one teacher teaches."""
    return [subject for subject, name in TEACHER_OF.items() if name == teacher]


def is_subject(name):
    """True for a known subject, ignoring capitals."""
    return name.capitalize() in SUBJECTS
