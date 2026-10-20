"""Kilometres and miles: reading, converting and showing distances and paces."""

from config import KM_PER_MILE


def parse_distance(text):
    """"5" or "5 km" -> 5.0 km; "3.1 mi" -> 4.99 km."""
    number, _, unit = text.strip().partition(" ")
    km = float(number)
    return km * KM_PER_MILE if unit.strip().lower() in ("mi", "mile", "miles") else km


def format_distance(km, unit):
    """12.345 km -> "12.3 km", or "7.7 mi" in miles."""
    shown = km / KM_PER_MILE if unit == "mi" else km
    return f"{shown:.1f} {unit}"


def format_pace(minutes_per_km, unit):
    """5.5 minutes per km -> "5:30 /km"; in miles the time per mile."""
    minutes = minutes_per_km * KM_PER_MILE if unit == "mi" else minutes_per_km
    whole, seconds = divmod(round(minutes * 60), 60)
    return f"{whole}:{seconds:02d} /{unit}"
