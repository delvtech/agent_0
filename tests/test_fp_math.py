"""fp.FixedPoint class math tests inspired from solidity hyperdrive implementation"""

import math
import unittest

import fixedpointmath as fp

# pylint: disable=unneeded-not


class TestFixedPointMath(unittest.TestCase):
    """Unit tests to verify that the fixed-point math implementations are correct."""

    APPROX_EQ = fp.FixedPoint(1e3)

    ONE = fp.FixedPoint("1.0")
    TWO = fp.FixedPoint("2.0")
    NEG_ONE = fp.FixedPoint("-1.0")
    INF = fp.FixedPoint("inf")
    NEG_INF = fp.FixedPoint("-inf")
    NAN = fp.FixedPoint("nan")

    def test_clip(self):
        """Test clip method with finite values."""
        assert fp.clip(0, 1, 2) == 1
        assert fp.clip(-1, -5, 5) == -1
        assert fp.clip(3, -3, -1) == -1
        assert fp.clip(-1.0, -3.0, 1) == -1.0
        assert fp.clip(1.0, 3.0, 3.0) == 3.0
        assert fp.clip(fp.FixedPoint(1.0), fp.FixedPoint(0.0), fp.FixedPoint(3.0)) == fp.FixedPoint(1.0)
        assert fp.clip(
            fp.FixedPoint(1.0), fp.FixedPoint(scaled_value=1), fp.FixedPoint(scaled_value=int(1e18 + 1))
        ) == fp.FixedPoint(1.0)

    def test_clip_nonfinite(self):
        """Test clip method with non-finite values."""
        assert fp.clip(self.NAN, self.NEG_ONE, self.ONE).is_nan() is True
        assert fp.clip(self.NAN, self.NEG_INF, self.INF).is_nan() is True
        assert fp.clip(self.ONE, self.NEG_INF, self.INF) == self.ONE
        assert fp.clip(self.ONE, self.NEG_INF, self.NEG_ONE) == self.NEG_ONE
        assert fp.clip(self.INF, self.NEG_INF, self.INF) == self.INF
        assert fp.clip(self.INF, self.NEG_INF, self.ONE) == self.ONE
        assert fp.clip(self.NEG_INF, self.NEG_ONE, self.INF) == self.NEG_ONE

    def test_clip_error(self):
        """Test clip method with bad inputs (min > max)."""
        with self.assertRaises(ValueError):
            _ = fp.clip(fp.FixedPoint(5.0), self.INF, self.NEG_INF)
        with self.assertRaises(ValueError):
            _ = fp.clip(5, 3, 1)

    def test_minimum(self):
        """Test minimum function."""
        assert fp.minimum(0, 1) == 0
        assert fp.minimum(-1, 1) == -1
        assert fp.minimum(-1, -3) == -3
        assert fp.minimum(-1.0, -3.0) == -3.0
        assert fp.minimum(1.0, 3.0) == 1.0
        assert fp.minimum(fp.FixedPoint(1.0), fp.FixedPoint(3.0)) == fp.FixedPoint(1.0)
        assert fp.minimum(fp.FixedPoint("3.0"), fp.FixedPoint(scaled_value=int(3e18 - 1e-17))) == fp.FixedPoint(
            scaled_value=int(3e18 - 1e-17)
        )

    def test_minimum_nonfinite(self):
        """Test minimum method."""
        assert fp.minimum(self.NAN, self.NEG_ONE).is_nan() is True
        assert fp.minimum(self.NAN, self.INF).is_nan() is True
        assert fp.minimum(self.ONE, self.INF) == self.ONE
        assert fp.minimum(self.NEG_ONE, self.NEG_INF) == self.NEG_INF
        assert fp.minimum(self.INF, self.NEG_INF) == self.NEG_INF

    def test_maximum(self):
        """Test maximum function."""
        assert fp.maximum(0, 1) == 1
        assert fp.maximum(-1, 1) == 1
        assert fp.maximum(-1, -3) == -1
        assert fp.maximum(-1.0, -3.0) == -1.0
        assert fp.maximum(1.0, 3.0) == 3.0
        assert fp.maximum(fp.FixedPoint(1.0), fp.FixedPoint(3.0)) == fp.FixedPoint(3.0)
        assert fp.maximum(fp.FixedPoint("3.0"), fp.FixedPoint(scaled_value=int(3e18 - 1e-17))) == fp.FixedPoint(3.0)

    def test_maximum_nonfinite(self):
        """Test maximum method."""
        assert fp.maximum(self.NAN, self.NEG_ONE).is_nan() is True
        assert fp.maximum(self.NAN, self.INF).is_nan() is True
        assert fp.maximum(self.ONE, self.INF) == self.INF
        assert fp.maximum(self.NEG_ONE, self.NEG_INF) == self.NEG_ONE
        assert fp.maximum(self.INF, self.NEG_INF) == self.INF

    def test_exp(self):
        """Test exp function."""
        tolerance = 1e-18
        result = fp.exp(fp.FixedPoint("1.0"))
        expected = fp.FixedPoint(scaled_value=2718281828459045235)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = fp.exp(fp.FixedPoint("-1.0"))
        expected = fp.FixedPoint(scaled_value=367879441171442321)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = fp.exp(1)
        expected = int(math.exp(1))
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = fp.exp(-1)
        expected = int(math.exp(-1))
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = fp.exp(1.0)
        expected = math.exp(1.0)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"
        result = fp.exp(-1.0)
        expected = math.exp(-1.0)
        assert math.isclose(result, expected, rel_tol=tolerance), f"exp(x):\n  {result=},\n{expected=}"

    def test_exp_nonfinite(self):
        """Test exp method."""
        assert fp.exp(self.NAN).is_nan() is True
        assert fp.exp(self.INF) == self.INF
        assert fp.exp(self.NEG_INF) == self.NEG_INF

    def test_sqrt(self):
        r"""Test sqrt method."""
        result = fp.sqrt(self.ONE)
        expected = self.ONE
        self.assertEqual(result, expected)
        result = fp.sqrt(fp.FixedPoint("5.0"))
        expected = fp.FixedPoint("2.236067977499789696")
        self.assertAlmostEqual(result, expected, delta=self.APPROX_EQ)
        result = fp.sqrt(3)
        expected = int(math.sqrt(3))
        self.assertEqual(result, expected)
        result = fp.sqrt(7.0)
        expected = float(math.sqrt(7.0))
        self.assertEqual(result, expected)

    def test_sqrt_nonfinite(self):
        r"""Test non-finite mode of fixed-point sqrt."""
        assert fp.sqrt(self.NAN).is_nan() is True
        assert fp.sqrt(self.INF) == self.INF

    def test_sqrt_fail(self):
        r"""Test failure mode of fixed-point sqrt."""
        with self.assertRaises(ValueError):
            _ = fp.sqrt(fp.FixedPoint("-inf"))

    def test_isclose(self):
        r"""Test fixed-point isclose method."""
        self.assertEqual(fp.isclose(self.ONE, self.ONE), True)
        self.assertEqual(fp.isclose(self.ONE, self.TWO), False)
        self.assertEqual(fp.isclose(self.ONE, self.TWO, abs_tol=self.ONE), True)
        delta = fp.FixedPoint("0.00001")
        self.assertEqual(fp.isclose(self.ONE, self.ONE + delta, abs_tol=delta), True)
        self.assertEqual(fp.isclose(self.ONE, self.ONE + delta, abs_tol=delta / 10), False)

    def test_isclose_nonfinite(self):
        r"""Test fixed-point isclose method for non-finite values."""
        self.assertEqual(fp.isclose(self.INF, self.INF), True)
        self.assertEqual(fp.isclose(self.NEG_INF, self.NEG_INF), True)
        self.assertEqual(fp.isclose(self.INF, self.NEG_INF), False)
        self.assertEqual(fp.isclose(self.NAN, self.NAN), False)
        self.assertEqual(fp.isclose(self.INF, self.NAN), False)
        self.assertEqual(fp.isclose(self.NEG_INF, self.NAN), False)

    def test_isclose_fail(self):
        r"""Test failure mode of fixed-point isclose."""
        with self.assertRaises(ValueError):
            _ = fp.isclose(fp.FixedPoint("5.0"), fp.FixedPoint("1.0"), abs_tol=fp.FixedPoint("inf"))
        with self.assertRaises(ValueError):
            _ = fp.isclose(fp.FixedPoint("5.0"), fp.FixedPoint("1.0"), abs_tol=fp.FixedPoint("-inf"))
        with self.assertRaises(ValueError):
            _ = fp.isclose(fp.FixedPoint("5.0"), fp.FixedPoint("1.0"), abs_tol=fp.FixedPoint("nan"))
