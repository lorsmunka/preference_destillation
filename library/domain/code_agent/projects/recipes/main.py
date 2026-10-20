"""Family meal planner."""

from cooking import cook
from meal_plan import plan_meal, print_week
from nutrition import week_report
from pantry import add_to_pantry
from ratings import family_favourite, rate
from search import can_cook_now, recipes_with
from shopping_list import print_shopping_list

HELP = "plan / week / shop / cook / pantry / find / ready / rate / calories / quit"


def run(command):
    if command == "plan":
        print(plan_meal(input("Day: ").strip().capitalize(), input("Recipe: ").strip(), int(input("Servings: "))) or "Planned.")
    elif command == "week":
        print_week()
    elif command == "shop":
        print_shopping_list()
    elif command == "cook":
        print(cook(input("Day: ").strip().capitalize()) or "Enjoy your meal!")
    elif command == "pantry":
        add_to_pantry(input("Ingredient: ").strip().lower(), float(input("Amount: ")), input("Unit: ").strip())
    elif command == "find":
        print(", ".join(recipes_with(input("Ingredient: ").strip())) or "Nothing found.")
    elif command == "ready":
        print(", ".join(can_cook_now()) or "Go shopping first.")
    elif command == "rate":
        rate(input("Recipe: ").strip(), input("Your name: ").strip(), int(input("Score 1-5: ")))
    elif command == "calories":
        for day, calories, share in week_report():
            print(f"{day:<10} {calories:>5} kcal ({share:.0%} of the day)")
    else:
        print(HELP)


def main():
    print(f"Family favourite: {family_favourite() or 'not decided yet'}")
    while True:
        command = input("\n> ").strip().lower()
        if command == "quit":
            break
        run(command)


if __name__ == "__main__":
    main()
