"""Significance tests for seed cohorts, standard library only (scipy is not a dependency).

    result = welch_t_test(values_a, values_b)       # t, degrees_of_freedom, p_value
    corrected = bonferroni(result.p_value, comparison_count)

Welch's t-test compares two means without assuming equal variances. The two-sided p-value
comes from the Student t distribution, through the regularized incomplete beta function.
Bonferroni: with m comparisons, each p-value is multiplied by m (capped at 1), so the chance
of any false positive across all m stays at the chosen level.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class WelchResult:
    t: float
    degrees_of_freedom: float
    p_value: float   # two-sided, uncorrected


def welch_t_test(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[WelchResult]:
    """Welch's two-sample t-test. None when a group has fewer than two values or neither
    group has any spread (the test is undefined then)."""
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    squared_error_a = statistics.variance(values_a) / len(values_a)
    squared_error_b = statistics.variance(values_b) / len(values_b)
    squared_error = squared_error_a + squared_error_b
    if squared_error == 0:
        return None
    t = (statistics.mean(values_a) - statistics.mean(values_b)) / math.sqrt(squared_error)
    degrees_of_freedom = squared_error ** 2 / (squared_error_a ** 2 / (len(values_a) - 1)
                                               + squared_error_b ** 2 / (len(values_b) - 1))
    return WelchResult(t=t, degrees_of_freedom=degrees_of_freedom,
                       p_value=student_t_two_sided_p(t, degrees_of_freedom))


def bonferroni(p_value: float, comparison_count: int) -> float:
    return min(1.0, p_value * comparison_count)


def student_t_two_sided_p(t: float, degrees_of_freedom: float) -> float:
    """P(|T| >= |t|) for a Student t variable with the given degrees of freedom."""
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    return regularized_incomplete_beta(x, degrees_of_freedom / 2, 0.5)


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b), by its continued fraction (Numerical Recipes, betai)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _beta_continued_fraction(x, a, b) / a
    return 1 - math.exp(log_front) * _beta_continued_fraction(1 - x, b, a) / b


def _beta_continued_fraction(x: float, a: float, b: float,
                             max_terms: int = 300, tolerance: float = 1e-14) -> float:
    """Modified Lentz evaluation of the incomplete-beta continued fraction."""
    smallest = 1e-300
    numerator_ratio = 1.0
    denominator_ratio = 1.0 - (a + b) * x / (a + 1)
    denominator_ratio = 1.0 / (denominator_ratio if abs(denominator_ratio) > smallest else smallest)
    fraction = denominator_ratio
    for term in range(1, max_terms + 1):
        for coefficient in (term * (b - term) * x / ((a + 2 * term - 1) * (a + 2 * term)),
                            -(a + term) * (a + b + term) * x / ((a + 2 * term) * (a + 2 * term + 1))):
            denominator_ratio = 1.0 + coefficient * denominator_ratio
            denominator_ratio = 1.0 / (denominator_ratio if abs(denominator_ratio) > smallest else smallest)
            numerator_ratio = 1.0 + coefficient / numerator_ratio
            numerator_ratio = numerator_ratio if abs(numerator_ratio) > smallest else smallest
            step = denominator_ratio * numerator_ratio
            fraction *= step
        if abs(step - 1.0) < tolerance:
            break
    return fraction
