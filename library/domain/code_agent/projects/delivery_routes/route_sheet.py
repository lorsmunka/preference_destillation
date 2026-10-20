"""The daily route sheet a driver takes along."""

import os

from config import DEPOT, SHEETS_FOLDER
from timing import clock_text


def stop_line(number, arrival, packages):
    """Time, place and the packages to drop there; "!" marks a heavy one."""
    here = [package.package_id + ("!" if package.is_heavy else "") for package in packages
            if package.address == arrival.place]
    late = f"  LATE by {arrival.late_by:.0f} min" if arrival.is_late else ""
    return f"{number:>2}. {clock_text(arrival.arrive)} {arrival.place:<18} {', '.join(here)}{late}"


def via_text(table, start, end):
    """The places passed between two stops, like "via Mill Lane > Bridge Street"."""
    middle = table.way_between(start, end)[1:-1]
    return "    via " + " > ".join(middle) if middle else "    direct"


def sheet_lines(plan, day, table):
    driver = plan.driver
    lines = [f"{day}  {driver.name}: {len(driver.cargo)} packages, {driver.load_kg:.0f} kg, {plan.total_km:.1f} km"]
    place = DEPOT
    for number, arrival in enumerate(plan.schedule, 1):
        lines += [via_text(table, place, arrival.place), stop_line(number, arrival, driver.cargo)]
        place = arrival.place
    lines.append(f"Back at the depot at {clock_text(plan.back_at)}, {plan.overtime:.0f} min overtime")
    return lines


def save_sheet(lines, day, driver_name):
    """Write the sheet to route_sheets/<day>_<driver>.txt and return that path."""
    os.makedirs(SHEETS_FOLDER, exist_ok=True)
    path = os.path.join(SHEETS_FOLDER, f"{day}_{driver_name.lower()}.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return path
