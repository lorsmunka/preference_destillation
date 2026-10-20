"""The alert thresholds: defaults, changed by a JSON config file."""

import json
import os

from config import DEFAULT_THRESHOLDS, THRESHOLDS_FILE


def load_thresholds(path=THRESHOLDS_FILE):
    """The defaults, with the values of the config file on top."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    if os.path.exists(path):
        with open(path) as file:
            thresholds.update(json.load(file))
    problems = threshold_problems(thresholds)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return thresholds


def threshold_problems(thresholds):
    """Settings that make no sense, like warning above critical."""
    problems = [f"unknown: {name}" for name in thresholds if name not in DEFAULT_THRESHOLDS]
    if not 0 <= thresholds["warning_error_rate"] <= thresholds["critical_error_rate"] <= 1:
        problems.append("error rates must be fractions, warning <= critical")
    if thresholds["slow_seconds"] <= 0:
        problems.append("slow_seconds must be above zero")
    return problems
