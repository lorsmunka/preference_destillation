"""Searching the catalogue and adding new books."""

from books import Book
from storage import load_books


def search_by_title(words):
    """Every book whose title contains `words`, ignoring capitals."""
    words = words.lower()
    return [(isbn, data["title"]) for isbn, data in load_books().items() if words in data["title"].lower()]


def search_by_author(name):
    name = name.lower()
    return [(isbn, data["title"]) for isbn, data in load_books().items() if name in data["author"].lower()]


def add_book(isbn, title, author, copies=1):
    """Put a new book in the catalogue with all its copies on the shelf."""
    if isbn in load_books():
        return None
    book = Book(isbn, title, author, copies, copies)
    book.save()
    return book


def print_results(results):
    if not results:
        print("No books found.")
    for isbn, title in results:
        print(f"  {isbn}  {title}")
