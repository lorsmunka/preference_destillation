"""Body weight: a moving average smooths out the daily ups and downs; its change is the trend."""

from config import MOVING_AVERAGE_DAYS, STEADY_CHANGE_KG


def moving_average(values, window=MOVING_AVERAGE_DAYS):
    """Each value averaged with up to window - 1 values before it: [70, 72, 71] -> [70, 71, 71]."""
    averages = []
    for index in range(len(values)):
        recent = values[max(0, index - window + 1):index + 1]
        averages.append(sum(recent) / len(recent))
    return averages


def weight_trend(log):
    """"losing", "gaining" or "steady": how the average moved over the last MOVING_AVERAGE_DAYS weigh-ins."""
    weights = [log.weights[day] for day in sorted(log.weights)]
    if len(weights) < 2:
        return "not enough weigh-ins"
    averages = moving_average(weights)
    change = averages[-1] - averages[-1 - min(MOVING_AVERAGE_DAYS, len(averages) - 1)]
    if abs(change) < STEADY_CHANGE_KG:
        return "steady"
    return "losing" if change < 0 else "gaining"
