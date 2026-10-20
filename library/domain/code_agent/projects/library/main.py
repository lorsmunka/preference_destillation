"""School library terminal: scan your member card, then borrow, return or reserve books."""

from books import find_book
from catalog import print_results, search_by_author, search_by_title
from loans import borrow_book, loans_of, return_book
from members import find_member
from menu import ask_choice, show_menu
from notifications import book_is_back
from reservations import reserve_book
from validation import valid_member_id


def scan_card():
    """Ask for a member card until a known one is typed."""
    while True:
        member_id = input("Member card: ").strip().upper()
        member = find_member(member_id) if valid_member_id(member_id) else None
        if member:
            return member
        print("Unknown card, try again.")


def handle(choice, member):
    if choice in ("1", "2"):
        words = input("Search for: ")
        print_results(search_by_title(words) if choice == "1" else search_by_author(words))
    elif choice in ("3", "4", "5"):
        book = find_book(input("ISBN: ").strip())
        if book is None:
            print("No such book.")
        elif choice == "3":
            print(borrow_book(member, book) or f"Enjoy {book.title}!")
        elif choice == "4":
            if return_book(member.member_id, book):
                book_is_back(book.isbn)
        else:
            print("Reserved." if reserve_book(member.member_id, book.isbn) else "You are already in line.")
    elif choice == "6":
        for loan in loans_of(member.member_id):
            print(f"  {loan['isbn']} due {loan['due']}")


def main():
    member = scan_card()
    while True:
        show_menu(member.name)
        choice = ask_choice()
        if choice == "0":
            break
        handle(choice, member)


if __name__ == "__main__":
    main()
