"""Shortest paths over the city map: Dijkstra's algorithm with a heap."""

import heapq


def dijkstra(city_map, start):
    """The shortest km from start to every place it can reach, and the place before each one."""
    shortest, previous = {start: 0}, {}
    heap, done = [(0, start)], set()
    while heap:
        distance, place = heapq.heappop(heap)
        if place in done:
            continue
        done.add(place)
        for neighbour, km in city_map.neighbours(place):
            if distance + km < shortest.get(neighbour, float("inf")):
                shortest[neighbour] = distance + km
                previous[neighbour] = place
                heapq.heappush(heap, (distance + km, neighbour))
    return shortest, previous


def path_to(previous, start, end):
    """The places from start to end: follow the previous places back from the end."""
    places = [end]
    while places[-1] != start:
        places.append(previous[places[-1]])
    return places[::-1]
