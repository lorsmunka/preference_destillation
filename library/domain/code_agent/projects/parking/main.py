"""Parking garage control: drive in, pay, drive out."""

from clock import hours_text, minutes_between, now
from gates import open_entry_gate, open_exit_gate
from passes import buy_pass
from payments import amount_due, pay
from pricing import price_table
from reports import occupancy
from tickets import find_ticket
from vehicles import Vehicle, valid_plate


def drive_in():
    plate = input("Number plate: ")
    if not valid_plate(plate):
        print("That is not a number plate.")
        return
    kind = input("car / motorbike / van: ").strip().lower()
    open_entry_gate(Vehicle(plate, kind))


def pay_machine():
    ticket = find_ticket(input("Ticket number: ").strip())
    if ticket is None:
        print("Unknown ticket.")
        return
    print(f"Parked {hours_text(minutes_between(ticket.entered, now()))}: {amount_due(ticket):.2f} to pay")
    change = pay(ticket, float(input("Money put in: ")))
    print("Not enough." if change is None else f"Paid. Change: {change}")


def main():
    for hours, fee in price_table():
        print(f"{hours:>2} h  {fee:6.2f}")
    while True:
        print(f"\nGarage {occupancy():.0%} full. (i)n, (p)ay, (o)ut, (m)onthly pass, (q)uit")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "i":
            drive_in()
        elif choice == "p":
            pay_machine()
        elif choice == "o":
            ticket = find_ticket(input("Ticket number: ").strip())
            if ticket:
                open_exit_gate(ticket)
        elif choice == "m":
            print(f"Pass sold: {buy_pass(input('Plate: '), input('Month (YYYY-MM): ')):.2f}")


if __name__ == "__main__":
    main()
