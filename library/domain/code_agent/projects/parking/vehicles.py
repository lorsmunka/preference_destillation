"""Vehicles that come in: a number plate and a kind."""

KINDS = ["car", "motorbike", "van"]


class Vehicle:
    def __init__(self, plate, kind="car"):
        self.plate = clean_plate(plate)
        self.kind = kind if kind in KINDS else "car"

    def is_motorbike(self):
        return self.kind == "motorbike"


def clean_plate(plate):
    """Upper case and no spaces or dashes: "ab-12 cd" -> "AB12CD"."""
    return plate.upper().replace(" ", "").replace("-", "")


def valid_plate(plate):
    """A plate has 5 to 8 letters and digits, and at least one of each."""
    plate = clean_plate(plate)
    has_letter = any(char.isalpha() for char in plate)
    has_digit = any(char.isdigit() for char in plate)
    return 5 <= len(plate) <= 8 and plate.isalnum() and has_letter and has_digit
