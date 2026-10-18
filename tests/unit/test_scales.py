"""The descriptor's block-rescaling rule."""

from __future__ import annotations

import dataclasses
import math

import pytest

from pihm.problems import ladder
from pihm.problems.scales import block_scales, vanishes
from pihm.problems.spec import Equation, Factor, Term


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("R1", 4.0),  # f' + 2f: 2 * 2
        ("R2a", 4.0),  # f'' + 4f' + 4f: 2 sqrt(4)
        ("R2b", 2 * math.sqrt(3.0)),  # f'' - 2f' - 3f
        ("R2c", 40.0),  # f'' + 5f' + 400f
        ("R4b", 4.0),
        ("R4c", 2 * math.sqrt(6.0)),
        ("R4d", 2 * math.sqrt(20.0)),
    ],
)
def test_a_regular_ode_takes_twice_its_characteristic_scale(name: str, expected: float) -> None:
    for n in (3, 7):
        assert block_scales(ladder.problem(name), [n]) == pytest.approx((expected,), rel=1e-12)


@pytest.mark.parametrize("name", ["R3a", "R3b", "R3c", "R3d", "R4a"])
def test_a_vanishing_leading_coefficient_takes_half_the_resolution(name: str) -> None:
    """R3's (1 - x^2) and (1 - x^2)^2, R4a's x - 1: lambda = N/2."""
    assert block_scales(ladder.problem(name), [5]) == (16.0,)
    assert block_scales(ladder.problem(name), [7]) == (64.0,)


def test_two_axes_take_the_resolution_balanced_across_second_order_terms() -> None:
    corrected = "corrected"
    assert block_scales(ladder.problem("R5a", corrected), [3, 4]) == (8.0, 16.0)
    assert block_scales(ladder.problem("R5b", corrected), [4, 4]) == (16.0, 16.0)
    assert block_scales(ladder.problem("R5c", corrected), [4, 4]) == (32.0, 16.0)  # (c N, N)


def test_a_missing_zeroth_order_term_uses_the_lowest_present() -> None:
    """The equation f'' + 3f' = 0 has rho = |a_1 / a_2| = 3, so lambda = 6."""
    problem = dataclasses.replace(
        ladder.problem("R2a"),
        equations=(Equation((Term(1.0, (Factor("f", (2,)),)), Term(3.0, (Factor("f", (1,)),)))),),
    )
    assert block_scales(problem, [4]) == pytest.approx((6.0,))


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([1.0, 0.0, -1.0], True),  # 1 - x^2: simple zeros at the ends
        ([1.0, 0.0, -2.0, 0.0, 1.0], True),  # (1 - x^2)^2: double zeros at the ends
        ([-1.0, 1.0], True),  # x - 1
        ([0.0, 1.0], True),  # x: an interior zero
        ([2.0, 1.0], False),  # 2 + x
        ([1.0, 0.0, 1.0], False),  # 1 + x^2: complex zeros
        ([-1.5, 1.0], False),  # x - 1.5: a zero outside
        ([3.0], False),
    ],
)
def test_vanishes_finds_zeros_on_the_interval(coefficients: list[float], expected: bool) -> None:
    assert vanishes(coefficients) is expected
