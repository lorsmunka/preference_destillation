"""Settings for the log analyzer (alert thresholds: see thresholds.json)."""

DEFAULT_LOG = "access.log"
THRESHOLDS_FILE = "thresholds.json"
TOP_COUNT = 10

DEFAULT_THRESHOLDS = {
    "warning_error_rate": 0.02,   # share of requests with a server error
    "critical_error_rate": 0.10,
    "slow_seconds": 1.5,          # for the 95th percentile
    "min_hour_requests": 20,      # quieter hours are not judged
}
