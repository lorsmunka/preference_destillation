"""Class gradebook: a teacher logs in, then enters scores, takes attendance and prints reports."""

from attendance import mark
from grades import record_grade
from homework import print_missing
from menu import ask_choice, show_menu
from rankings import print_ranking
from report_cards import print_report_card
from students import add_student, students_in_class
from teachers import class_of, teacher_login
from validation import read_score


def enter_score(class_name):
    subject = input("Subject: ").strip().capitalize()
    for student in students_in_class(class_name):
        score = read_score(input(f"{student.full_name()}: "))
        if score is not None:
            print(record_grade(student.student_id, subject, score) or "saved")


def take_attendance(class_name):
    for student in students_in_class(class_name):
        answer = input(f"{student.full_name()} here? (y/n) ").strip().lower()
        mark(student.student_id, answer == "y")


def main():
    initials = teacher_login()
    if initials is None:
        return
    class_name = class_of(initials)
    while True:
        show_menu(class_name)
        choice = ask_choice()
        if choice == "q":
            break
        if choice == "1":
            enter_score(class_name)
        elif choice == "2":
            take_attendance(class_name)
        elif choice == "3":
            for student in students_in_class(class_name):
                print_report_card(student)
        elif choice == "4":
            print_ranking(class_name)
        elif choice == "5":
            print_missing(class_name, input("Subject: ").strip().capitalize(), int(input("Homeworks set: ")))
        elif choice == "6":
            add_student(input("First name: "), input("Last name: "), class_name)


if __name__ == "__main__":
    main()
