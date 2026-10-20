"""The ActivityLog: every workout, the steps of each day and the body weights, kept in a JSON file."""

import json
import os
from dataclasses import asdict

from config import DATA_FILE, REFERENCE_WEIGHT_KG
from workouts import Workout, calories_burned


class ActivityLog:
    def __init__(self, workouts=None, steps=None, weights=None):
        self.workouts = workouts or []
        self.steps = steps or {}       # day -> steps walked that day
        self.weights = weights or {}   # day -> body weight in kg

    def add_workout(self, day, exercise, minutes, km=0.0):
        """Log a workout; its calories are worked out with the latest body weight."""
        calories = calories_burned(exercise, minutes, self.latest_weight())
        workout = Workout(day, exercise, minutes, km, calories)
        self.workouts.append(workout)
        return workout

    def add_steps(self, day, count):
        """Steps add up: 3000 in the morning and 5000 later make 8000 for the day."""
        self.steps[day] = self.steps.get(day, 0) + count

    def record_weight(self, day, kg):
        self.weights[day] = kg

    def latest_weight(self):
        return self.weights[max(self.weights)] if self.weights else REFERENCE_WEIGHT_KG

    def workouts_on(self, days):
        return [workout for workout in self.workouts if workout.day in days]

    def save(self):
        data = {"workouts": [asdict(workout) for workout in self.workouts], "steps": self.steps,
                "weights": self.weights}
        with open(DATA_FILE, "w") as file:
            json.dump(data, file, indent=1)


def load_log():
    """The saved ActivityLog, or an empty one the first time."""
    if not os.path.exists(DATA_FILE):
        return ActivityLog()
    with open(DATA_FILE) as file:
        data = json.load(file)
    return ActivityLog([Workout(**workout) for workout in data["workouts"]], data["steps"], data["weights"])
