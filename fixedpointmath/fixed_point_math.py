"""Math functions that support FixedPoint number format."""

from __future__ import annotations

import math
from typing import TypeVar

from .fixed_point import FixedPoint
from .fixed_point_integer_math import FixedPointIntegerMath

NUMERIC = TypeVar("NUMERIC", FixedPoint, int, float)

# we will use single letter names for these functions since they do basic arithmetic
# pylint: disable=invalid-name


def clip(x: NUMERIC, low: NUMERIC, high: NUMERIC) -> NUMERIC:
    """Clip the input, x, to be within (min, max), inclusive"""
    if low > high:
        raise ValueError(f"{low=} must be <= {high=}.")
    return minimum(maximum(x, low), high)


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


def maximum(*args: NUMERIC) -> NUMERIC:
    """Compare the inputs and return the greatest value.

    If the first argument equals the second, return the first.
    """
    # use builtin for generic types
    if isinstance(args[0], (float, int)):
        return type(args[0])(max(*args))
    # else, we're FixedPoint
    current_max = FixedPoint("-inf")
    for arg in args:
        if isinstance(arg, FixedPoint) and arg.is_nan():  # any nan means minimum is nan
            return arg
        if arg >= current_max:  # pylint: disable=consider-using-max-builtin
            current_max = arg
    return type(args[0])(current_max)


def minimum(*args: NUMERIC) -> NUMERIC:
    """Compare the inputs and return the lowest value.

    If the first argument equals the second, return the first.
    """
    # use builtin for generic types
    if isinstance(args[0], (int, float)):
        return type(args[0])(min(*args))
    # else, we're FixedPoint
    current_min = FixedPoint("inf")
    for arg in args:
        if isinstance(arg, FixedPoint) and arg.is_nan():  # any nan means minimum is nan
            return arg
        if arg <= current_min:  # pylint: disable=consider-using-min-builtin
            current_min = arg
    return type(args[0])(current_min)


def sqrt(x: NUMERIC) -> NUMERIC:
    """Performs sqrt(x)"""
    return type(x)(math.sqrt(x))
