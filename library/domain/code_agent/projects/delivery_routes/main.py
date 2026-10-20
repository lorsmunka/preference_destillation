"""Plan a day of deliveries: python main.py [--day 2026-10-21] [--driver Anna]"""

import argparse
from datetime import date

from city_map import load_map
from distances import DistanceTable
from drivers import load_drivers
from packages import load_packages
from planner import plan_day
from route_sheet import save_sheet, sheet_lines


def read_arguments():
    parser = argparse.ArgumentParser(description="Plan the delivery routes of one day.")
    parser.add_argument("--day", default=date.today().isoformat())
    parser.add_argument("--driver", help="print only this driver's sheet")
    return parser.parse_args()


def main():
    arguments = read_arguments()
    city_map = load_map()
    table = DistanceTable(city_map)
    packages, unknown = load_packages(city_map)
    plans, left_over = plan_day(table, packages, load_drivers())
    for package in unknown + left_over:
        print(f"Not planned: {package.package_id} for {package.address} ({package.weight_kg} kg)")
    for plan in plans:
        print(f"{plan.driver.name:<10} {len(plan.order):>2} stops {plan.total_km:6.1f} km {len(plan.late_stops)} late")
        lines = sheet_lines(plan, arguments.day, table)
        if arguments.driver in (None, plan.driver.name):
            print("\n".join(lines))
        print("Saved as", save_sheet(lines, arguments.day, plan.driver.name))


if __name__ == "__main__":
    main()
