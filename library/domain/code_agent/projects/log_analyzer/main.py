"""Log analyzer: read an access log, filter it, report on it."""

import sys

from alerts import check_alerts
from cli import argument_problem, build_parser
from filters import filter_from_args
from parsing import read_entries
from summary import build_report
from thresholds import load_thresholds
from writers import make_writer


def run(args):
    """Write the report; exit code 1 on a critical alert."""
    entries, bad_lines = read_entries(args.log_file)
    entries = filter_from_args(args).apply(entries)
    alerts = check_alerts(entries, load_thresholds(args.limits_file))
    report = build_report(entries, bad_lines, alerts, args.top)
    if args.output:
        with open(args.output, "w", newline="") as file:
            make_writer(args.output_format, file).write_report(report)
        print(f"Report written to {args.output}")
    else:
        make_writer(args.output_format, sys.stdout).write_report(report)
    return 1 if any(alert.level == "critical" for alert in alerts) else 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    problem = argument_problem(args)
    if problem:
        parser.error(problem)
    try:
        code = run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.exit(code)


if __name__ == "__main__":
    main()
