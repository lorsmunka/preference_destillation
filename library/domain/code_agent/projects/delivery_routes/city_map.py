"""The city map: places joined by streets, each street with its length in km."""

import json

from config import MAP_FILE


class CityMap:
    def __init__(self):
        self.streets = {}   # place -> {neighbour: km}

    def add_street(self, start, end, km, one_way=False):
        """Most streets can be driven both ways; a one-way street only from start to end."""
        self.streets.setdefault(start, {})[end] = km
        self.streets.setdefault(end, {})
        if not one_way:
            self.streets[end][start] = km

    def neighbours(self, place):
        return self.streets.get(place, {}).items()

    def has_place(self, place):
        return place in self.streets


def load_map(path=MAP_FILE):
    """The file lists streets: {"from": "Depot", "to": "Oak Street", "km": 1.2, "one_way": false}."""
    city_map = CityMap()
    with open(path, encoding="utf-8") as file:
        for street in json.load(file):
            city_map.add_street(street["from"], street["to"], street["km"], street.get("one_way", False))
    return city_map
