"""Hotel front desk."""

from bookings import cancel_booking, make_booking
from checkin import check_in, check_out
from guests import find_or_add_guest, is_regular
from housekeeping import rooms_to_clean
from pricing import stay_price
from rooms import free_rooms, get_room
from validation import check_stay, read_day, valid_email


def new_booking():
    arrival, departure = read_day(input("Arrival: ")), read_day(input("Departure: "))
    problem = check_stay(arrival, departure)
    if problem:
        print(problem)
        return
    people = int(input("People: "))
    options = free_rooms(arrival, departure, people)
    if not options:
        print("Sorry, we are full.")
        return
    for number in options:
        print(f"  room {number}: {stay_price(get_room(number).kind, arrival, departure, people):.2f}")
    email = input("Email: ")
    if not valid_email(email):
        print("That email does not look right.")
        return
    guest = find_or_add_guest(input("Name: "), email)
    if is_regular(guest):
        print("Welcome back!")
    print(f"Booking {make_booking(guest, input('Room: ').strip(), arrival, departure, people)}")


def main():
    while True:
        command = input("\n(b)ook, (c)ancel, check (i)n, check (o)ut, (h)ousekeeping, (q)uit: ").strip().lower()
        if command == "q":
            break
        if command == "b":
            new_booking()
        elif command == "c":
            print(f"Cancelled, fee {cancel_booking(int(input('Booking number: '))):.2f}")
        elif command == "i":
            print(f"Key for room {check_in(int(input('Booking number: ')))}")
        elif command == "o":
            print(f"Bill: {check_out(int(input('Booking number: ')), float(input('Extras: ') or 0)):.2f}")
        elif command == "h":
            print("To clean: " + ", ".join(rooms_to_clean()))


if __name__ == "__main__":
    main()
