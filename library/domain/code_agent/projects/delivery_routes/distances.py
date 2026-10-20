"""Distances between places, with one Dijkstra run per start place, remembered for the day."""

from routing import dijkstra, path_to


class DistanceTable:
    def __init__(self, city_map):
        self.city_map = city_map
        self.runs = {}   # start place -> (distances, previous)

    def run_from(self, start):
        if start not in self.runs:
            self.runs[start] = dijkstra(self.city_map, start)
        return self.runs[start]

    def km_between(self, start, end):
        """The shortest distance; infinite when end cannot be reached."""
        return self.run_from(start)[0].get(end, float("inf"))

    def way_between(self, start, end):
        """The places passed on the shortest way from start to end."""
        return path_to(self.run_from(start)[1], start, end)

    def nearest(self, start, places):
        return min(places, key=lambda place: self.km_between(start, place))
