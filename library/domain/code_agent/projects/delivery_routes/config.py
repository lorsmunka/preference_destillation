"""Settings for the delivery planner."""

MAP_FILE = "data/city_map.json"
PACKAGES_FILE = "data/packages.json"
DRIVERS_FILE = "data/drivers.json"
SHEETS_FOLDER = "route_sheets"

DEPOT = "Depot"
MAX_STOPS = 25
MINUTES_PER_STOP = 4
HEAVY_PACKAGE_KG = 20

AVERAGE_SPEED_KMH = 30
RUSH_HOURS = [(7, 9, 1.6), (12, 13, 1.2), (16, 18, 1.5)]   # (first hour, last hour, slow-down)
QUIET_FACTOR = 0.9
