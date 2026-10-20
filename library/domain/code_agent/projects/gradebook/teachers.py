"""Teachers log in with their initials and a password from teachers.json."""

from storage import load_teachers


def check_password(initials, password):
    """True when the teacher exists and the password matches."""
    teacher = load_teachers().get(initials.upper())
    return teacher is not None and teacher["password"] == password


def teacher_login(tries=3):
    """Ask for initials and password, at most `tries` times. The teacher's initials, or None."""
    for _ in range(tries):
        initials = input("Initials: ").strip().upper()
        password = input("Password: ")
        if check_password(initials, password):
            return initials
        print("That did not match.")
    return None


def class_of(initials):
    """The class this teacher is the form tutor of."""
    return load_teachers()[initials]["class"]
