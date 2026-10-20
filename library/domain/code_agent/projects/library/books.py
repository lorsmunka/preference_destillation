"""A Book in the catalogue, with how many copies the library owns and how many are on the shelf."""

from storage import load_books, save_books


class Book:
    def __init__(self, isbn, title, author, copies, on_shelf):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.copies = copies
        self.on_shelf = on_shelf

    def is_available(self):
        return self.on_shelf > 0

    def take_copy(self):
        """One copy leaves the shelf. False when none is left."""
        if not self.is_available():
            return False
        self.on_shelf -= 1
        return True

    def put_back(self):
        if self.on_shelf < self.copies:
            self.on_shelf += 1

    def save(self):
        books = load_books()
        books[self.isbn] = {"title": self.title, "author": self.author,
                            "copies": self.copies, "on_shelf": self.on_shelf}
        save_books(books)


def find_book(isbn):
    """The Book with this ISBN, or None."""
    data = load_books().get(isbn)
    if data is None:
        return None
    return Book(isbn, data["title"], data["author"], data["copies"], data["on_shelf"])
