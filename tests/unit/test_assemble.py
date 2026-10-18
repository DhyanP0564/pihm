"""The standard-form assembler, checked construction by construction against the paper."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from pihm import basis
from pihm.assemble import assemble
from pihm.classical import reference_state
from pihm.operators import (
    collapse_row,
    derivative_matrix,
    embedding,
    fold,
    multiplication,
    power_of_x,
    zero_row,
)
from pihm.problems import ladder
from pihm.problems.functions import maclaurin_coefficients
from pihm.problems.reference import reference_solution, regular_value


def _dense(matrix: sp.csr_array) -> np.ndarray:
    return np.asarray(matrix.toarray())


def _derivative(n: int) -> np.ndarray:
    return _dense(derivative_matrix(n))


@pytest.mark.parametrize("name", ["R2a", "R2b", "R2c"])
def test_fig3_is_square_as_printed(name: str) -> None:
    """Fig. 3: A = aG^2 + bG + c1 on n qubits and B_n(x_z) or B_n(x_m) G (Eqs 15 and 16)."""
    n = 3
    residual = assemble(ladder.problem(name), n)
    assert residual.lifted == (False,)
    a, b, c = [term.coefficient for term in ladder.problem(name).equations[0].terms]
    G = _derivative(n)
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), a * G @ G + b * G + c * np.eye(8))
    (constraint,) = ladder.problem(name).constraints
    expected = _dense(zero_row(constraint.at, n)) @ np.linalg.matrix_power(G, constraint.derivative)
    np.testing.assert_allclose(_dense(residual.blocks[1].matrix), expected)


def test_fig4_is_lifted_as_printed() -> None:
    """Fig. 4, m = 0: A = (M_1 - M_{x^2}) G^2 - 2 M_x G + l(l+1) M_1 and B_{n+1}(0) M_1 G."""
    n = 3
    M = {p: _dense(power_of_x(p, n)) for p in range(3)}
    G = _derivative(n)
    residual = assemble(ladder.problem("R3a"), n)
    assert residual.lifted == (True,)
    expected = (M[0] - M[2]) @ G @ G - 2 * M[1] @ G + 20 * M[0]
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), expected, atol=1e-12)
    row = _dense(zero_row(0.0, n + 1)) @ _dense(embedding(n)) @ G
    np.testing.assert_allclose(_dense(residual.blocks[1].matrix), row, atol=1e-14)


def test_fig4_associated_is_lifted_as_printed() -> None:
    """Fig. 4, m = 1: (M_1 - 2M_{x^2} + M_{x^4}) G^2 + 2(M_{x^3} - M_x) G - 30 M_{x^2} + 29 M_1."""
    n = 3
    M = {p: _dense(power_of_x(p, n)) for p in range(5)}
    G = _derivative(n)
    expected = (M[0] - 2 * M[2] + M[4]) @ G @ G + 2 * (M[3] - M[1]) @ G - 30 * M[2] + 29 * M[0]
    residual = assemble(ladder.problem("R3c", "faithful"), n)
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), expected, atol=1e-11)


def test_fig5a_injects_its_source_through_the_collapse() -> None:
    """Fig. 5a: A = (M_x - M_1)G^2 - M_x G + M_1 - (M_{x^2} - 2M_x + M_1) D0(x_s)."""
    n = 3
    problem = ladder.problem("R4a", "faithful")
    M = {p: _dense(power_of_x(p, n)) for p in range(3)}
    G = _derivative(n)
    D = _dense(collapse_row(0.0, regular_value(problem), n))
    expected = (M[1] - M[0]) @ G @ G - M[1] @ G + M[0] - (M[2] - 2 * M[1] + M[0]) @ D
    residual = assemble(problem, n)
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), expected, atol=1e-12)
    row = _dense(zero_row(-0.2, n + 1)) @ _dense(embedding(n)) @ G
    np.testing.assert_allclose(_dense(residual.blocks[1].matrix), row, atol=1e-14)


def test_fig5d_maclaurin_source_is_eq14() -> None:
    """Fig. 5d: A = M_1(G^2 + 4G + 20) - (sum_{p <= 7} c_p M_{x^p}) D0(x_s)."""
    n = 3
    problem = ladder.problem("R4d", "faithful")
    G = _derivative(n)
    series = maclaurin_coefficients(lambda z: z * np.exp(-2 * z) * np.sin(4 * z), 7)
    source = sum(c * _dense(power_of_x(p, n)) for p, c in enumerate(series))
    D = _dense(collapse_row(0.0, -1.0, n))
    expected = _dense(embedding(n)) @ (G @ G + 4 * G + 20 * np.eye(8)) - source @ D
    residual = assemble(problem, n)
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), expected, atol=1e-11)


def test_fig6_uses_kronecker_sums_and_slices() -> None:
    """Fig. 6: Laplace A = G^2 (+) G^2; heat A = G (x) 1 - k 1 (x) G^2; slices B (x) 1, 1 (x) B."""
    n = 2
    G, I4 = _derivative(n), np.eye(4)
    laplace = assemble(ladder.problem("R5a", "faithful"), n)
    np.testing.assert_allclose(
        _dense(laplace.blocks[0].matrix), np.kron(G @ G, I4) + np.kron(I4, G @ G)
    )
    B = {x: _dense(zero_row(x, n)) for x in (1.0, -1.0)}
    np.testing.assert_allclose(_dense(laplace.blocks[1].matrix), np.kron(B[1.0], I4))
    np.testing.assert_allclose(_dense(laplace.blocks[3].matrix), np.kron(I4, B[-1.0]))
    heat = assemble(ladder.problem("R5b", "faithful"), n)
    np.testing.assert_allclose(
        _dense(heat.blocks[0].matrix), np.kron(G, I4) - np.kron(I4, G @ G) / 25
    )
    slope = _dense(zero_row(0.75, n)) @ G
    np.testing.assert_allclose(_dense(heat.blocks[3].matrix), np.kron(I4, slope))


def test_fig7_doubled_space_as_printed() -> None:
    """Fig. 7b: A = N_1[D0 (x) (4G^2 + 1) + 2 G (x) G]; the invariant is N_1(D0 (x) B_n G)."""
    n = 2
    G = _derivative(n)
    N1 = _dense(fold(2, n))
    D = _dense(collapse_row(0.0, 1.0, n))
    residual = assemble(ladder.problem("R6a"), n)
    assert residual.nonlinearity == "doubled"
    expected = N1 @ (np.kron(D, 4 * G @ G + np.eye(4)) + 2 * np.kron(G, G))
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), expected, atol=1e-12)
    constraint = N1 @ np.kron(D, _dense(zero_row(0.0, n)) @ G)
    np.testing.assert_allclose(_dense(residual.blocks[1].matrix), constraint, atol=1e-14)


def test_fig7c_folds_its_source_with_nx() -> None:
    """Fig. 7c: A = N_1(D0 (x) G^2 - 2 1 (x) 1) + N_x(D0 (x) D0)."""
    n = 2
    problem = ladder.problem("R6b", "faithful")
    G = _derivative(n)
    D = _dense(collapse_row(0.5, regular_value(problem), n))
    expected = _dense(fold(2, n)) @ (np.kron(D, G @ G) - 2 * np.eye(16)) + _dense(
        fold(2, n, weight="x")
    ) @ np.kron(D, D)
    np.testing.assert_allclose(_dense(assemble(problem, n).blocks[0].matrix), expected, atol=1e-12)


def test_exact_product_target_is_a_null_vector() -> None:
    """R6: Eq. 29's polynomial solution gives an exact kernel product state."""
    for n in (2, 3, 4):
        residual = assemble(ladder.problem("R6a"), n)
        b = reference_solution(residual.problem).coefficients(n)
        target = np.kron(b, b) / np.linalg.norm(b) ** 2
        assert np.linalg.norm(residual.matrix() @ target) < 1e-13


def test_square_mode_uses_the_galerkin_truncation() -> None:
    n = 3
    residual = assemble(ladder.problem("R3a"), n, mult="square")
    assert residual.lifted == (False,)
    G = _derivative(n)
    square = _dense(multiplication((1.0, 0.0, -1.0), n, mode="square"))
    x = _dense(multiplication((0.0, -2.0), n, mode="square"))
    expected = square @ G @ G + x @ G + 20 * np.eye(8)
    np.testing.assert_allclose(_dense(residual.blocks[0].matrix), expected, atol=1e-12)


def test_compaction_keeps_the_spectrum() -> None:
    residual = assemble(ladder.problem("R5b"), 3)
    full, compact = residual.matrix(compact=False), residual.matrix()
    assert compact.shape[0] < full.shape[0]
    np.testing.assert_allclose(
        reference_state(compact).singular_values,
        reference_state(full).singular_values,
        rtol=1e-10,
        atol=1e-12,
    )


def test_tau_truncation_leaves_an_exact_kernel() -> None:
    """Dropping the top k equation rows makes the kernel exact (the tau method)."""
    residual = assemble(ladder.problem("R2a"), 4, truncation="tau")
    assert residual.blocks[0].matrix.shape == (14, 16)
    state = reference_state(residual.matrix())
    assert state.singular_values[0] < 1e-13 * state.singular_values[-1]
    assert state.trusted
    with pytest.raises(ValueError, match="square one-axis"):
        assemble(ladder.problem("R3a"), 4, truncation="tau")


def test_datum_rows_vanish_on_the_truncated_reference() -> None:
    residual = assemble(ladder.problem("R5b"), 5)
    b = reference_solution(residual.problem).coefficients(5)
    (datum,) = [block for block in residual.blocks if block.name.startswith("datum")]
    assert np.max(np.abs(datum.matrix @ b)) < 1e-8 * np.max(np.abs(b))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"regime": "descriptor", "truncation": "tau"}, ValueError, "ablation"),
        ({"regime": "other"}, ValueError, "is not one of"),
        ({"nonlinearity": "doubled"}, ValueError, "is linear"),
        ({"mult": "wide"}, ValueError, "is not one of"),
    ],
)
def test_options_are_checked(kwargs: dict[str, str], error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        assemble(ladder.problem("R2a"), 3, **kwargs)  # type: ignore[arg-type]


def test_resolution_is_checked() -> None:
    with pytest.raises(ValueError, match="one qubit count"):
        assemble(ladder.problem("R5a"), (2, 2, 2))
    with pytest.raises(ValueError, match="one qubit count"):
        assemble(ladder.problem("R2a"), 0)
    with pytest.raises(ValueError, match="quadratic"):
        assemble(ladder.problem("R6a"), 2, nonlinearity="none")
    with pytest.raises(ValueError, match="mult='lift'"):
        assemble(ladder.problem("R6a"), 2, mult="square")


def test_unequal_axes_and_the_readout() -> None:
    residual = assemble(ladder.problem("R5c"), (3, 2))
    assert residual.dimension == 32
    assert residual.readout.qubits == (3, 2)
    assert residual.readout.power == 2
    doubled = assemble(ladder.problem("R6a"), 2)
    assert doubled.readout.qubits == (3,)
    assert doubled.readout.power == 1
    b = basis.fit(lambda x: 1 - x**2 / 8, 2)
    decoded = doubled.readout.decoder @ np.kron(b, b)  # f_q = f / ||b||^2 ... times ||b||^2
    np.testing.assert_allclose(decoded, basis.fit(lambda x: 1 - x**2 / 8, 3), atol=1e-14)
    tweaked = dataclasses.replace(ladder.problem("R2a"), description="other")
    assert assemble(tweaked, 2).problem is tweaked
