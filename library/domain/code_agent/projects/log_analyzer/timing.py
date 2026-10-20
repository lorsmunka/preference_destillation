"""Response times: the percentiles of how long the answers took."""

import math

PERCENTILES = (50, 90, 95, 99)


def percentile(values, percent):
    """The smallest value with `percent` % of the values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def response_percentiles(entries):
    """{50: seconds, 90: seconds, ...} over all entries."""
    seconds = [entry.seconds for entry in entries]
    return {percent: percentile(seconds, percent) for percent in PERCENTILES}
