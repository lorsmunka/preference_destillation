"""Arrival times along a route: driving with traffic, waiting for time windows, time at each stop."""

from dataclasses import dataclass

from config import DEPOT, MINUTES_PER_STOP
from timing import drive_minutes


@dataclass
class Arrival:
    place: str
    arrive: float   # minutes after midnight
    leave: float
    late_by: float

    @property
    def is_late(self):
        return self.late_by > 0


def stop_windows(packages):
    """Place -> (start, end); with several packages for one place, the time all of them allow."""
    windows = {}
    for package in packages:
        start, end = windows.get(package.address, (0, 24 * 60))
        windows[package.address] = (max(start, package.window_start), min(end, package.window_end))
    return windows


def estimate_arrivals(order, driver, table):
    """Leave the depot when the shift starts; wait at a stop that is reached too early."""
    windows, arrivals = stop_windows(driver.cargo), []
    place, now = DEPOT, driver.shift_start
    for stop in order:
        now += drive_minutes(table.km_between(place, stop), now)
        start, end = windows[stop]
        arrivals.append(Arrival(stop, now, max(now, start) + MINUTES_PER_STOP, max(0, now - end)))
        place, now = stop, arrivals[-1].leave
    return arrivals


def back_at_depot(arrivals, table, shift_start):
    """When the van is back at the depot after the last stop."""
    if not arrivals:
        return shift_start
    last = arrivals[-1]
    return last.leave + drive_minutes(table.km_between(last.place, DEPOT), last.leave)
