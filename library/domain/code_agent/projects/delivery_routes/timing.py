"""Times of day as minutes after midnight ("09:30" is 570), and driving time with traffic."""

from config import AVERAGE_SPEED_KMH, QUIET_FACTOR, RUSH_HOURS


def to_minutes(text):
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


def clock_text(minute):
    minute = round(minute)
    return f"{minute // 60 % 24:02d}:{minute % 60:02d}"


def traffic_factor(minute):
    """How many times longer than normal a drive takes when it starts at this minute."""
    hour = int(minute // 60) % 24
    for first, last, factor in RUSH_HOURS:
        if first <= hour <= last:
            return factor
    return QUIET_FACTOR if hour < 6 or hour >= 20 else 1.0


def drive_minutes(km, depart_minute):
    """Minutes for km at the average speed, times the traffic factor when leaving."""
    return km / AVERAGE_SPEED_KMH * 60 * traffic_factor(depart_minute)
