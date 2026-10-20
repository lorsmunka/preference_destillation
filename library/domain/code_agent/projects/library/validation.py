"""Checks for what people type in: ISBNs and member cards."""


def valid_isbn(isbn):
    """An ISBN-13: 13 digits whose check digit works out."""
    digits = isbn.replace("-", "")
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(int(digit) * (1 if index % 2 == 0 else 3) for index, digit in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == int(digits[12])


def valid_member_id(member_id):
    """Member cards look like "M" and four digits, for example M0042."""
    return len(member_id) == 5 and member_id[0] == "M" and member_id[1:].isdigit()


def clean_title(title):
    """A title with extra spaces removed and every word capitalised."""
    return " ".join(word.capitalize() for word in title.split())
