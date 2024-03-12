"""FixedPoint class math tests inspired from solidity hyperdrive implementation"""

import math
import unittest

from fixedpointmath import FixedPoint, clip, exp, isclose, maximum, minimum, sqrt

# pylint: disable=unneeded-not


class TestFixedPointMath(unittest.TestCase):
    """Unit tests to verify that the fixed-point math implementations are correct."""

    APPROX_EQ = FixedPoint(1e3)

    ONE = FixedPoint("1.0")
    TWO = FixedPoint("2.0")
    NEG_ONE = FixedPoint("-1.0")
    INF = FixedPoint("inf")
    NEG_INF = FixedPoint("-inf")
    NAN = FixedPoint("nan")

    def test_clip(self):
        """Test clip method with finite values."""
        assert clip(0, 1, 2) == FixedPoint("1")
        assert clip(-1, -5, 5) == FixedPoint("-1")
        assert clip(3, -3, -1) == FixedPoint("-1")
        assert clip(-1.0, -3.0, 1) == FixedPoint("-1.0")
        assert clip(1.0, 3.0, 3.0) == FixedPoint("3.0")
        assert clip(FixedPoint(1.0), FixedPoint(0.0), FixedPoint(3.0)) == FixedPoint(1.0)
        assert clip(FixedPoint(1.0), FixedPoint(scaled_value=1), FixedPoint(scaled_value=int(1e18 + 1))) == FixedPoint(
            1.0
        )

    def test_clip_nonfinite(self):
        """Test clip method with non-finite values."""
        assert clip(self.NAN, self.NEG_ONE, self.ONE).is_nan() is True
        assert clip(self.NAN, self.NEG_INF, self.INF).is_nan() is True
        assert clip(self.ONE, self.NEG_INF, self.INF) == self.ONE
        assert clip(self.ONE, self.NEG_INF, self.NEG_ONE) == self.NEG_ONE
        assert clip(self.INF, self.NEG_INF, self.INF) == self.INF
        assert clip(self.INF, self.NEG_INF, self.ONE) == self.ONE
        assert clip(self.NEG_INF, self.NEG_ONE, self.INF) == self.NEG_ONE

    def test_clip_error(self):
        """Test clip method with bad inputs (min > max)."""
        with self.assertRaises(ValueError):
            _ = clip(FixedPoint(5.0), self.INF, self.NEG_INF)
        with self.assertRaises(ValueError):
            _ = clip(5, 3, 1)

    def test_minimum(self):
        """Test minimum function."""
        assert minimum(0, 1) == FixedPoint("0")
        assert minimum(-1, 1) == FixedPoint("-1")
        assert minimum(-1, -3) == FixedPoint("-3")
        assert minimum(-1, 0, -3) == FixedPoint("-3")
        assert minimum(-1.0, -3.0) == FixedPoint("-3.0")
        assert minimum(1.0, 3.0) == FixedPoint("1.0")
        assert minimum(1.0, 3.0, 0.5) == FixedPoint("0.5")
        assert minimum(FixedPoint("1.0"), FixedPoint("3.0")) == FixedPoint("1.0")
        assert minimum(FixedPoint("3.0"), FixedPoint(scaled_value=int(3e18 - 1e-17))) == FixedPoint(
            scaled_value=int(3e18 - 1e-17)
        )
        assert minimum(FixedPoint("1.0"), FixedPoint("-100.0"), FixedPoint("3.0")) == FixedPoint("-100.0")

    def test_minimum_nonfinite(self):
        """Test minimum method."""
        assert minimum(self.NAN, self.NEG_ONE).is_nan() is True
        assert minimum(self.NAN, self.INF).is_nan() is True
        assert minimum(self.ONE, self.INF) == self.ONE
        assert minimum(self.NEG_ONE, self.NEG_INF) == self.NEG_INF
        assert minimum(self.INF, self.NEG_INF) == self.NEG_INF

    def test_maximum(self):
        """Test maximum function."""
        assert maximum(0, 1) == FixedPoint("1")
        assert maximum(-1, 1) == FixedPoint("1")
        assert maximum(-1, -3) == FixedPoint("-1")
        assert maximum(-1, 0, -3) == FixedPoint("0")
        assert maximum(-1.0, 0.0, -3.0) == FixedPoint("0.0")
        assert maximum(1.0, 3.0) == FixedPoint("3.0")
        assert maximum(FixedPoint("1.0"), FixedPoint("3.0")) == FixedPoint("3.0")
        assert maximum(FixedPoint("1.0"), FixedPoint("100.0"), FixedPoint("3.0")) == FixedPoint("100")
        assert maximum(FixedPoint("3.0"), FixedPoint(scaled_value=int(3e18 - 1e-17))) == FixedPoint(3.0)

    def test_maximum_nonfinite(self):
        """Test maximum method."""
        assert maximum(self.NAN, self.NEG_ONE).is_nan() is True
        assert maximum(self.NAN, self.INF).is_nan() is True
        assert maximum(self.ONE, self.INF) == self.INF
        assert maximum(self.NEG_ONE, self.NEG_INF) == self.NEG_ONE
        assert maximum(self.INF, self.NEG_INF) == self.INF

    def test_exp(self):
        """Test exp function."""
        tolerance = 1e-18
        result = exp(FixedPoint("1.0"))
        expected = FixedPoint(scaled_value=2718281828459045235)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = exp(FixedPoint("-1.0"))
        expected = FixedPoint(scaled_value=367879441171442321)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = exp(1)
        expected = int(math.exp(1))
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = exp(-1)
        expected = int(math.exp(-1))
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = exp(1.0)
        expected = math.exp(1.0)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = exp(-1.0)
        expected = math.exp(-1.0)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"

    def test_exp_nonfinite(self):
        """Test exp method."""
        assert exp(self.NAN).is_nan() is True
        assert exp(self.INF) == self.INF
        assert exp(self.NEG_INF) == self.NEG_INF

    def test_sqrt(self):
        r"""Test sqrt method."""
        result = sqrt(self.ONE)
        expected = self.ONE
        self.assertEqual(result, expected)
        result = sqrt(FixedPoint("5.0"))
        expected = FixedPoint("2.236067977499789696")
        self.assertAlmostEqual(result, expected, delta=self.APPROX_EQ)
        result = sqrt(3)
        expected = int(math.sqrt(3))
        self.assertEqual(result, expected)
        result = sqrt(7.0)
        expected = float(math.sqrt(7.0))
        self.assertEqual(result, expected)

    def test_sqrt_nonfinite(self):
        r"""Test non-finite mode of fixed-point sqrt."""
        assert sqrt(self.NAN).is_nan() is True
        assert sqrt(self.INF) == self.INF

    def test_sqrt_fail(self):
        r"""Test failure mode of fixed-point sqrt."""
        with self.assertRaises(ValueError):
            _ = sqrt(FixedPoint("-inf"))

    def test_isclose(self):
        r"""Test fixed-point isclose method."""
        self.assertEqual(isclose(self.ONE, self.ONE), True)
        self.assertEqual(isclose(self.ONE, self.TWO), False)
        self.assertEqual(isclose(self.ONE, self.TWO, abs_tol=self.ONE), True)
        delta = FixedPoint("0.00001")
        self.assertEqual(isclose(self.ONE, self.ONE + delta, abs_tol=delta), True)
        self.assertEqual(isclose(self.ONE, self.ONE + delta, abs_tol=delta / 10), False)

    def test_isclose_nonfinite(self):
        r"""Test fixed-point isclose method for non-finite values."""
        self.assertEqual(isclose(self.INF, self.INF), True)
        self.assertEqual(isclose(self.NEG_INF, self.NEG_INF), True)
        self.assertEqual(isclose(self.INF, self.NEG_INF), False)
        self.assertEqual(isclose(self.NAN, self.NAN), False)
        self.assertEqual(isclose(self.INF, self.NAN), False)
        self.assertEqual(isclose(self.NEG_INF, self.NAN), False)

    def test_isclose_fail(self):
        r"""Test failure mode of fixed-point isclose."""
        with self.assertRaises(ValueError):
            _ = isclose(FixedPoint("5.0"), FixedPoint("1.0"), abs_tol=FixedPoint("inf"))
        with self.assertRaises(ValueError):
            _ = isclose(FixedPoint("5.0"), FixedPoint("1.0"), abs_tol=FixedPoint("-inf"))
        with self.assertRaises(ValueError):
            _ = isclose(FixedPoint("5.0"), FixedPoint("1.0"), abs_tol=FixedPoint("nan"))
