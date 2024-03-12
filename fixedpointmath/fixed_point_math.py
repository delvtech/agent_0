"""Math functions that support FixedPoint number format."""

from __future__ import annotations

import math
from typing import TypeVar

from .fixed_point import FixedPoint
from .fixed_point_integer_math import FixedPointIntegerMath

NUMERIC = TypeVar("NUMERIC", FixedPoint, int, float)

# we will use single letter names for these functions since they do basic arithmetic
# pylint: disable=invalid-name


def exp(x: NUMERIC) -> NUMERIC:
    """Performs e^x"""
    if isinstance(x, FixedPoint):
        if not x.isfinite():
            return x
        return FixedPoint(scaled_value=FixedPointIntegerMath.exp(x.scaled_value))
    return type(x)(math.exp(x))


def isclose(a: NUMERIC, b: NUMERIC, abs_tol: NUMERIC = FixedPoint("0.0")) -> bool:
    """Checks `abs(a-b) <= abs_tol`.
    Ignores relative tolerance since FixedPoint should be accurate regardless of scale.

    Arguments
    ---------
    a: FixedPoint | int | float
        The first number to compare
    b: FixedPoint | int | float
        The second number to compare
    abs_tol: FixedPoint | int | float, optional
        The absolute tolerance.
        Defaults to zero, requiring a and b to be exact.
        Must be finite.

    Returns
    -------
    bool
        Whether or not the numbers are within the absolute tolerance.
    """
    # If a or b is inf then they need to be equal
    equality_conditions = [
        isinstance(a, FixedPoint) and not a.isfinite(),
        isinstance(b, FixedPoint) and not b.isfinite(),
        isinstance(a, float) and not math.isfinite(a),
        isinstance(b, float) and not math.isfinite(b),
    ]
    if any(equality_conditions):
        return a == b
    if (isinstance(abs_tol, FixedPoint) and not abs_tol.isfinite()) or (
        isinstance(abs_tol, float) and not math.isfinite(abs_tol)
    ):
        raise ValueError("Input abs_tol must be finite.")
    return abs(a - b) <= abs_tol


def maximum(*args: NUMERIC) -> FixedPoint:
    """Compare the inputs and return the greatest value.
    If inputs are not FixedPoint type, then we convert it to FixedPoint.
    """
    current_max = FixedPoint("-inf")
    for arg in args:
        arg = FixedPoint(arg)
        if arg.is_nan():  # any nan means minimum is nan
            return arg
        if arg >= current_max:  # pylint: disable=consider-using-max-builtin
            current_max = arg
    return current_max


def minimum(*args: NUMERIC) -> FixedPoint:
    """Compare the inputs and return the lowest value.
    If inputs are not FixedPoint type, then we convert it to FixedPoint.
    """
    current_min = FixedPoint("inf")
    for arg in args:
        arg = FixedPoint(arg)
        if arg.is_nan():  # any nan means minimum is nan
            return arg
        if arg <= current_min:  # pylint: disable=consider-using-min-builtin
            current_min = arg
    return current_min


def clip(x: NUMERIC, low: NUMERIC, high: NUMERIC) -> FixedPoint:
    """Clip the input, x, to be within (min, max), inclusive.
    If inputs are not FixedPoint type, then we convert it to FixedPoint.
    """
    if low > high:
        raise ValueError(f"{low=} must be <= {high=}.")
    return minimum(type(x)(maximum(x, low)), high)


def sqrt(x: NUMERIC) -> NUMERIC:
    """Performs sqrt(x)"""
    return type(x)(math.sqrt(x))
