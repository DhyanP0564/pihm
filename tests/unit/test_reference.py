"""Reference solutions: realisation, derivatives, exact roots and checks against the problem."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.special import lpmv

from pihm.problems import ladder
from pihm.problems.reference import (
    check_reference,
    exact_root,
    reference_solution,
    regular_value,
    slice_data,
)
from pihm.problems.spec import Regular

#: Every problem, then every panel of the paper as printed.
ALL = [(name, "corrected") for name in ladder.names()] + [
    (name, "faithful") for name in ladder.PANELS
]

#: Panels whose printed constraint position is rounded, so it is not a root of the solution
#:; the relative residual of the constraint on the true solution, at least.
ROUNDED = {"R4a": 5e-3, "R4b": 3e-3, "R4c": 1e-2, "R4d": 5e-2, "R6b": 1e-6}


@pytest.mark.parametrize(("name", "variant"), ALL)
@pytest.mark.nightly
def test_every_reference_satisfies_its_problem(name: str, variant: str) -> None:
    """The transcription check that the plan wanted from sympy, done numerically.

    Every equation residual is small against the terms' size, every stated datum holds, and
    every constraint vanishes except the faithful variants' rounded positions, to rounding
    or, for an initial-value reference's ratio, to ten times its tolerance. The equation's bound
    is rounding for closed forms, the solver's tolerance through the fitted series' derivative
    for the boundary-value and initial-value references, and the finite-difference error at a
    branch point.
    """
    problem = ladder.problem(name, variant)
    check = check_reference(problem)
    if problem.reference.kind == "bvp":
        bound = 1e-8
    elif problem.reference.kind == "ivp":
        bound = 1e-9
    else:
        bound = 1e-12 if reference_solution(problem).smooth else 1e-5
    assert check.equation < bound
    assert check.data < 1e-11
    for residual in check.constraints:
        if variant == "faithful" and name in ROUNDED:
            assert residual > ROUNDED[name]
        else:
            ivp = problem.reference.kind == "ivp"
            assert residual < (10 * problem.reference.tolerance if ivp else 1e-13)


def test_the_papers_laplace_solution_fails_the_check() -> None:
    """The printed csch(pi) cosh(pi x/2) sinh(pi(y+1)/2) is not even harmonic."""
    problem = ladder.problem("R5a", "faithful")
    printed = dataclasses.replace(
        problem,
        reference=dataclasses.replace(
            problem.reference, expression="cosh(pi*x/2)*sinh(pi*(y + 1)/2)/sinh(pi)"
        ),
    )
    check = check_reference(printed)
    assert check.equation > 0.5
    assert check.data > 0.1


@pytest.mark.parametrize(
    ("name", "near", "order", "root"),
    [
        ("R4a", -0.2, 1, -0.195976071956462),
        ("R4b", -0.73, 0, 1 - np.sqrt(3)),
        ("R4d", -0.75, 1, -0.7468053836480),
        ("R6b", 0.02615, 0, 0.0261470434329),
    ],
)
def test_exact_roots(name: str, near: float, order: int, root: float) -> None:
    assert exact_root(ladder.problem(name), near, order) == pytest.approx(root, abs=1e-12)


def test_exact_root_needs_a_sign_change() -> None:
    with pytest.raises(ValueError, match="no root"):
        exact_root(ladder.problem("R1"), 0.0)


def test_regular_value_prefers_the_stated_datum() -> None:
    assert regular_value(ladder.problem("R4d", "faithful")) == -1.0  # f(0) = -1, as printed
    printed_heat = ladder.problem("R5b", "faithful")
    assert regular_value(printed_heat) == pytest.approx(1.0)  # sin(2 pi / 4) on t = 0
    heat = ladder.problem("R2b")  # x_s = 0.5 is not a data point: the reference is used
    assert regular_value(heat) == pytest.approx(0.25 * (np.exp(1.5) - np.exp(-0.5)))
    corrected = ladder.problem("R3c")  # the solved v = f / sqrt(1 - x^2)
    assert regular_value(corrected) == pytest.approx(lpmv(1, 5, 0.5) / np.sqrt(0.75))
    with pytest.raises(ValueError, match="has value"):
        regular_value(dataclasses.replace(heat, regular=Regular("f", (0.0,))))


def test_smooth_series_are_cut_and_branch_points_detected() -> None:
    smooth = reference_solution(ladder.problem("R2a"))
    assert smooth.smooth
    assert smooth.series.size < 40
    branch = reference_solution(ladder.problem("R3c", "faithful"))
    assert not branch.smooth
    x = np.array([-0.9, 0.0, 0.4])
    # f' of P_5^1 by finite differences of the exact function, against the identity's derivative.
    h = 1e-6
    expected = (lpmv(1, 5, x + h) - lpmv(1, 5, x - h)) / (2 * h)
    np.testing.assert_allclose(branch.derivative((1,))(x), expected, rtol=1e-7)
    with pytest.raises(ValueError, match="orders 1 and 2"):
        branch.derivative((3,))(x)


def test_the_boundary_value_reference() -> None:
    reference = reference_solution(ladder.problem("R6b"))
    assert reference.physical(np.array([-1.0, 1.0])) == pytest.approx([-0.1, 0.1], abs=1e-12)
    assert reference.smooth


def test_coefficients_converge_for_the_branch_point() -> None:
    """Figs 4c and 4d alias at 64N nodes; the reference uses 2^17 nodes and converges."""
    b = reference_solution(ladder.problem("R3c", "faithful")).coefficients(5)
    assert float(b @ b) == pytest.approx(108.51142, abs=5e-6)
    b = reference_solution(ladder.problem("R3d", "faithful")).coefficients(5)
    assert float(b @ b) == pytest.approx(135.38360, abs=5e-6)


def test_a_systems_references_are_per_field() -> None:
    problem = ladder.problem("R7")
    x = np.linspace(-1.0, 1.0, 11)
    u, v = (reference_solution(problem, name) for name in ("u", "v"))
    np.testing.assert_allclose(u.field(x), np.sin(np.pi * (x + 1)), atol=1e-15)
    np.testing.assert_allclose(v.field(x), np.pi * np.cos(np.pi * (x + 1)), atol=1e-14)
    assert reference_solution(problem).field(0.3) == u.field(0.3)
    assert regular_value(problem) == pytest.approx(np.pi, rel=1e-15)
    with pytest.raises(ValueError, match="no field"):
        reference_solution(problem, "w")


def test_slice_data_reads_every_fields_value_at_the_point() -> None:
    np.testing.assert_allclose(slice_data(ladder.problem("R7"), "x", -1.0), [0.0, np.pi])
    np.testing.assert_allclose(slice_data(ladder.problem("R8"), "t", -1.0), [0.5])
    with pytest.raises(ValueError, match="states no value"):
        slice_data(ladder.problem("R7"), "x", 0.0)
