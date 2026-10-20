"""A Workout: what was done, for how long and how far, and the calories it burned."""

from dataclasses import dataclass

from config import REFERENCE_WEIGHT_KG

# calories per minute for a person of REFERENCE_WEIGHT_KG
CALORIES_PER_MINUTE = {"running": 11.0, "cycling": 8.0, "swimming": 9.5, "walking": 4.5, "rowing": 8.5,
                       "yoga": 3.0}


@dataclass
class Workout:
    day: str           # "YYYY-MM-DD"
    exercise: str
    minutes: int
    km: float = 0.0
    calories: float = 0.0

    @property
    def pace(self):
        """Minutes per km, or None for a workout without a distance."""
        return self.minutes / self.km if self.km else None


def calories_burned(exercise, minutes, body_weight):
    """The calories of the table, scaled by body weight: a heavier person burns more."""
    return round(CALORIES_PER_MINUTE[exercise] * minutes * body_weight / REFERENCE_WEIGHT_KG)
