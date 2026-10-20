"""Alerts: too many errors, or too slow answers."""

from dataclasses import dataclass

from counting import group_by_hour
from timing import percentile


@dataclass
class Alert:
    level: str   # "warning" or "critical"
    message: str


def error_rate(entries):
    """The share of requests with a server error (status 500 or more)."""
    if not entries:
        return 0.0
    return sum(entry.is_server_error for entry in entries) / len(entries)


def rate_alert(label, rate, thresholds):
    """A critical or warning Alert when the rate reaches its threshold."""
    for level in ("critical", "warning"):
        if rate >= thresholds[level + "_error_rate"]:
            return Alert(level, f"{label}: {rate:.1%} of requests failed")
    return None


def check_alerts(entries, thresholds):
    """The overall error rate, error spikes in busy hours, slow answers."""
    alerts = [rate_alert("all requests", error_rate(entries), thresholds)]
    for hour, group in group_by_hour(entries).items():
        if len(group) >= thresholds["min_hour_requests"]:
            alerts.append(rate_alert(f"hour {hour}", error_rate(group), thresholds))
    slow = percentile([entry.seconds for entry in entries], 95)
    if slow is not None and slow > thresholds["slow_seconds"]:
        alerts.append(Alert("warning", f"95% of answers took up to {slow:.2f} s"))
    return [alert for alert in alerts if alert is not None]
