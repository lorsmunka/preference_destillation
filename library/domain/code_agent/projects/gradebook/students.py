"""The pupils of a class."""

from storage import load_students, save_students


class Student:
    def __init__(self, student_id, first_name, last_name, class_name):
        self.student_id = student_id
        self.first_name = first_name
        self.last_name = last_name
        self.class_name = class_name

    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def save(self):
        students = load_students()
        students[self.student_id] = {"first": self.first_name, "last": self.last_name, "class": self.class_name}
        save_students(students)


def add_student(first_name, last_name, class_name):
    """Give a new pupil the next free number, like S0007, and save them."""
    students = load_students()
    student_id = f"S{len(students) + 1:04d}"
    student = Student(student_id, first_name, last_name, class_name)
    student.save()
    return student


def students_in_class(class_name):
    """All pupils of one class, sorted by last name."""
    found = [Student(sid, data["first"], data["last"], data["class"])
             for sid, data in load_students().items() if data["class"] == class_name]
    return sorted(found, key=lambda student: student.last_name)
