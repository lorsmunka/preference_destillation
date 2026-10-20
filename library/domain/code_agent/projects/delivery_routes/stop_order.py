"""The order of a driver's stops: nearest neighbour first, then 2-opt to remove detours."""

from config import DEPOT


def nearest_neighbour(stops, table):
    """From the depot, keep driving to the closest stop not visited yet."""
    order, current, left = [], DEPOT, list(stops)
    while left:
        current = table.nearest(current, left)
        left.remove(current)
        order.append(current)
    return order


def round_trip_km(order, table):
    """From the depot through every stop in order and back to the depot."""
    places = [DEPOT] + order + [DEPOT]
    return sum(table.km_between(places[index], places[index + 1]) for index in range(len(places) - 1))


def two_opt(order, table):
    """Turn a piece of the route around as long as that makes the round trip shorter."""
    best, improved = list(order), True
    while improved:
        improved = False
        for first in range(len(best) - 1):
            for last in range(first + 1, len(best)):
                candidate = best[:first] + best[first:last + 1][::-1] + best[last + 1:]
                if round_trip_km(candidate, table) < round_trip_km(best, table) - 0.001:
                    best, improved = candidate, True
    return best
