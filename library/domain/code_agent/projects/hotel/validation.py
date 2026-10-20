"""Checks on what is typed at the desk."""

from datetime import date


def read_day(text):
    """A date typed as YYYY-MM-DD, or None."""
    try:
        return date.fromisoformat(text.strip()).isoformat()
    except ValueError:
        return None


def valid_email(email):
    """Something@something.something, with no spaces."""
    email = email.strip()
    name, _, domain = email.partition("@")
    return bool(name) and "." in domain and " " not in email


def check_stay(arrival, departure):
    """None for a stay in the future that ends after it starts, otherwise what is wrong."""
    if arrival is None or departure is None:
        return "Please type dates as YYYY-MM-DD."
    if arrival < date.today().isoformat():
        return "That day has passed."
    if departure <= arrival:
        return "The departure must be after the arrival."
    return None
