"""Giving each package to a driver whose van has room and whose other stops are close by."""

from config import DEPOT


def detour_km(driver, package, table):
    """How far the package's address is from the driver's nearest stop (the depot for an empty van)."""
    places = driver.stops() or [DEPOT]
    return min(table.km_between(place, package.address) for place in places)


def assignment_order(packages):
    """The hardest packages first: the narrowest time window, then the heaviest."""
    return sorted(packages, key=lambda package: (package.window_length(), -package.weight_kg))


def assign_packages(packages, drivers, table):
    """Each package to the closest van that can take it (the emptier van on a tie). Returns the rest."""
    left_over = []
    for package in assignment_order(packages):
        candidates = [driver for driver in drivers if driver.can_take(package)]
        if not candidates:
            left_over.append(package)
            continue
        best = min(candidates, key=lambda driver: (detour_km(driver, package, table), driver.load_kg))
        best.cargo.append(package)
    return left_over
