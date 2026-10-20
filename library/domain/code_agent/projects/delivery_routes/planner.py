"""Planning the day: assign the packages, order each driver's stops, estimate the arrivals."""

from dataclasses import dataclass

from arrivals import back_at_depot, estimate_arrivals
from assignment import assign_packages
from drivers import Driver
from stop_order import nearest_neighbour, round_trip_km, two_opt


@dataclass
class RoutePlan:
    driver: Driver
    order: list
    schedule: list
    total_km: float
    back_at: float   # minutes after midnight

    @property
    def late_stops(self):
        return [arrival for arrival in self.schedule if arrival.is_late]

    @property
    def overtime(self):
        """Minutes the driver is back after the end of the shift (0 when in time)."""
        return max(0, self.back_at - self.driver.shift_end)


def plan_route(driver, table):
    order = two_opt(nearest_neighbour(driver.stops(), table), table)
    arrivals = estimate_arrivals(order, driver, table)
    back_at = back_at_depot(arrivals, table, driver.shift_start)
    return RoutePlan(driver, order, arrivals, round_trip_km(order, table), back_at)


def plan_day(table, packages, drivers):
    """A plan for each driver who got packages, and the packages no van could take."""
    left_over = assign_packages(packages, drivers, table)
    return [plan_route(driver, table) for driver in drivers if driver.cargo], left_over
