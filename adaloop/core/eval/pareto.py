"""Pareto front of (cost, accuracy) points."""
from collections.abc import Sequence


def pareto_front(costs: Sequence[float], accuracies: Sequence[float]) -> list[bool]:
    """Per point, True when no other point is at most as costly and at least as accurate, and better in one of the two.

    Lower cost and higher accuracy are better; identical points are on the front together.
    """
    points = list(zip(costs, accuracies, strict=True))
    return [not any(c <= cost and a >= acc and (c < cost or a > acc) for c, a in points) for cost, acc in points]
