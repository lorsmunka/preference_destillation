"""Drivers, their shifts and their vans."""

import json
from dataclasses import dataclass, field

from config import DRIVERS_FILE, MAX_STOPS
from timing import to_minutes


@dataclass
class Driver:
    name: str
    capacity_kg: float
    shift_start: int   # minutes after midnight
    shift_end: int
    cargo: list = field(default_factory=list)

    @property
    def load_kg(self):
        return sum(package.weight_kg for package in self.cargo)

    def can_take(self, package):
        """Room in the van, a stop to spare, and a time window that falls in the shift."""
        has_room = self.load_kg + package.weight_kg <= self.capacity_kg
        new_stop = package.address not in self.stops()
        if not has_room or (new_stop and len(self.stops()) >= MAX_STOPS):
            return False
        return package.window_start < self.shift_end and package.window_end > self.shift_start

    def stops(self):
        """The addresses to visit, each once: two packages for one house are one stop."""
        return list(dict.fromkeys(package.address for package in self.cargo))


def load_drivers(path=DRIVERS_FILE):
    with open(path, encoding="utf-8") as file:
        return [Driver(data["name"], data["van_kg"], to_minutes(data["start"]), to_minutes(data["end"]))
                for data in json.load(file)]
