"""To-do list in the terminal."""

from dates import read_date
from export import export_tasks, task_line
from priorities import sort_tasks
from reminders import reminder_text
from search import search_titles
from stats import print_stats
from storage import load_tasks
from tasks import Task, add_task, delete_task, finish_task
from users import login_screen
from validation import read_number, read_priority

COMMANDS = "a=add  d=done  x=delete  s=search  t=stats  e=export  q=quit"


def show_list(tasks):
    for number, task in enumerate(sort_tasks(tasks), 1):
        print(task_line(number, task))


def new_task(username):
    title = input("What needs doing? ")
    due = read_date(input("Due (today, tomorrow, YYYY-MM-DD or Enter): ") or "none")
    priority = read_priority(input("Priority (h/m/l): ")) or "medium"
    print(add_task(username, Task(title, due, priority)) or "Added.")


def main():
    username = login_screen()
    if username is None:
        print("Could not log in.")
        return
    while True:
        tasks = load_tasks(username)
        print("\n" + reminder_text(tasks))
        show_list(tasks)
        command = input(COMMANDS + "\n> ").strip().lower()
        if command == "q":
            break
        if command == "a":
            new_task(username)
        elif command in ("d", "x"):
            number = read_number(input("Task number: "), len(tasks))
            if number and command == "d":
                finish_task(username, number)
            elif number:
                print("Deleted " + delete_task(username, number))
        elif command == "s":
            show_list(search_titles(tasks, input("Words: ")))
        elif command == "t":
            print_stats(tasks)
        elif command == "e":
            print(f"Wrote {export_tasks(tasks)} tasks.")


if __name__ == "__main__":
    main()
