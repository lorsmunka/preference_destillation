"""Packages to deliver today: where they go, how heavy they are, when someone is home."""

import json
from dataclasses import dataclass

from config import HEAVY_PACKAGE_KG, PACKAGES_FILE
from timing import to_minutes


@dataclass
class Package:
    package_id: str
    address: str        # a place on the city map
    weight_kg: float
    window_start: int   # minutes after midnight
    window_end: int

    @property
    def is_heavy(self):
        return self.weight_kg >= HEAVY_PACKAGE_KG

    def window_length(self):
        return self.window_end - self.window_start


def load_packages(city_map, path=PACKAGES_FILE):
    """Today's packages, and apart from them the ones whose address is not on the map."""
    packages, unknown = [], []
    with open(path, encoding="utf-8") as file:
        for data in json.load(file):
            package = Package(data["id"], data["address"], data["kg"], to_minutes(data["from"]), to_minutes(data["until"]))
            if city_map.has_place(package.address):
                packages.append(package)
            else:
                unknown.append(package)
    return packages, unknown
