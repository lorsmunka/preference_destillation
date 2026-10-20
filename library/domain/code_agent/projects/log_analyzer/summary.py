"""Putting the numbers together: a Report section per question."""

from alerts import error_rate
from counting import group_by_hour, status_counts, top_ips, top_paths
from formatting import bar, format_seconds, format_size
from report import Report
from timing import response_percentiles


def overview_rows(entries, bad_lines):
    rows = [["requests", len(entries)], ["unreadable lines", bad_lines],
            ["data sent", format_size(sum(entry.size for entry in entries))],
            ["server errors", f"{error_rate(entries):.1%}"]]
    counts = status_counts(entries)
    return rows + [[f"status {name}", number] for name, number in counts.items()]


def hour_rows(entries):
    """Requests per hour, their server errors, and a bar to compare."""
    groups = group_by_hour(entries)
    busiest = max((len(group) for group in groups.values()), default=0)
    rows = []
    for hour, group in groups.items():
        errors = sum(entry.is_server_error for entry in group)
        rows.append([hour, len(group), errors, bar(len(group), busiest)])
    return rows


def build_report(entries, bad_lines, alerts, top):
    report = Report(f"Access log report: {len(entries)} requests")
    add = report.add_section
    add("Overview", ["measure", "value"], overview_rows(entries, bad_lines))
    add("Requests per hour", ["hour", "requests", "errors", ""], hour_rows(entries))
    add("Top paths", ["path", "requests"], top_paths(entries, top))
    add("Top IP addresses", ["address", "requests"], top_ips(entries, top))
    times = response_percentiles(entries)
    add("Response times", ["percentile", "time"],
        [[f"p{percent}", format_seconds(value)] for percent, value in times.items()])
    add("Alerts", ["level", "message"], [[alert.level, alert.message] for alert in alerts])
    return report
