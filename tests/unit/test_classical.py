"""The SVD reference solver and the safeguards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

from pihm import basis, classical
from pihm.assemble import assemble
from pihm.classical import (
    field_error,
    iterative_reference_state,
    lift_errors,
    physical_reference,
    readout_bound,
    reference_state,
    solve,
)
from pihm.operators import derivative_matrix, zero_row
from pihm.problems.ladder import problem

RECORDED = Path(__file__).resolve().parents[1] / "data" / "recorded_svd.npz"


def _with_spectrum(sigma: list[float], rng: np.random.Generator, rows: int = 12) -> np.ndarray:
    """Return a random rows x len(sigma) matrix with the given singular values."""
    columns = len(sigma)
    Q1, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
    Q2, _ = np.linalg.qr(rng.normal(size=(columns, columns)))
    S = np.zeros((rows, columns))
    S[:columns, :columns] = np.diag(sigma)
    return np.asarray(Q1 @ S @ Q2.T)


def _standard_residual(n: int, coefficients: tuple[float, float, float], row: Any) -> np.ndarray:
    """Return the standard-form residual [a G^2 + b G + c I; row] of a constant-coefficient ODE."""
    G = derivative_matrix(n)
    a, b, c = coefficients
    A = a * (G @ G) + b * G + c * sp.eye_array(2**n)
    return np.asarray(sp.vstack([A, row]).toarray())


def test_ground_state_and_spectrum(rng: np.random.Generator) -> None:
    sigma = [4.0, 3.0, 2.0, 1.0, 1e-8]
    state = reference_state(_with_spectrum(sigma, rng))
    # An SVD is accurate to about eps * sigma_max in absolute terms, so tiny values get an
    # absolute tolerance (a relative one fails across LAPACK builds).
    np.testing.assert_allclose(state.singular_values, sorted(sigma), rtol=1e-12, atol=1e-14)
    assert state.lambda0 == pytest.approx(1e-16, rel=1e-5)
    assert state.lambda1 == pytest.approx(1.0)
    assert state.gap == pytest.approx(1.0)
    assert state.hamiltonian_norm == pytest.approx(16.0)
    assert state.relative_gap == pytest.approx(1 / 16)
    assert state.kernel_dimension == 1
    assert state.trusted
    assert state.backward_error < 1e-14
    assert np.linalg.norm(state.vector) == pytest.approx(1.0)


def test_phase_is_fixed(rng: np.random.Generator) -> None:
    """The largest component is real and positive, so results do not flip sign between runs."""
    R = _with_spectrum([3.0, 2.0, 1.0, 1e-9], rng)
    for matrix in (R, -R, R.astype(np.complex128) * np.exp(0.7j)):
        v = reference_state(matrix).vector
        peak = v[np.argmax(np.abs(v))]
        assert peak.real > 0
        assert abs(peak.imag) < 1e-12


def test_complex_residual(rng: np.random.Generator) -> None:
    R = rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))
    null = rng.normal(size=4) + 1j * rng.normal(size=4)
    R = R - np.outer(R @ null, null.conj()) / np.vdot(null, null)
    state = reference_state(R)
    assert abs(np.vdot(state.vector, null)) / np.linalg.norm(null) == pytest.approx(1.0)
    assert state.backward_error < 1e-14


def test_wide_residual_has_a_kernel_of_the_right_size(rng: np.random.Generator) -> None:
    R = rng.normal(size=(3, 5))
    state = reference_state(R)
    assert state.kernel_dimension == 2
    np.testing.assert_allclose(state.singular_values[:2], 0.0)
    assert np.linalg.norm(R @ state.vector) < 1e-13


def test_kernel_dimension_two(rng: np.random.Generator) -> None:
    state = reference_state(_with_spectrum([4.0, 3.0, 2.0, 1.0, 1e-12, 1e-12], rng))
    assert state.kernel_dimension == 2
    assert not state.trusted


@pytest.mark.gotcha
def test_a_kernel_found_by_luck_is_not_trusted(rng: np.random.Generator) -> None:
    """A kernel of dimension 1 from a spectrum with no clean separation is not trusted."""
    state = reference_state(_with_spectrum([1.0, 0.9, 0.8, 0.7, 0.2], rng))
    assert state.kernel_dimension == 1
    assert state.gap_ratio == pytest.approx(3.5)
    assert not state.trusted


@pytest.mark.gotcha
@pytest.mark.parametrize(
    ("n", "trusted"), [(2, False), (3, False), (4, False), (5, True), (6, True)]
)
def test_under_resolved_stiff_problem_is_not_trusted(n: int, trusted: bool) -> None:
    """A real problem: Fig. 3d ({1,5,400}, f'(0) = 0) resolves only from n = 5.

    Below that the smallest singular values are all of order 10^2 and no kernel is defined
    (measured gap ratios 1.1, 1.5 and 9.0 at n = 2, 3, 4; 330 at n = 5).
    """
    R = _standard_residual(n, (1.0, 5.0, 400.0), zero_row(0.0, n) @ derivative_matrix(n))
    assert reference_state(R).trusted is trusted


@pytest.mark.gotcha
@pytest.mark.parametrize(("ratio", "trusted"), [(15.0, False), (19.7, False), (25.0, True)])
def test_a_kernel_must_clear_the_largest_ordinary_spacing(
    ratio: float, trusted: bool, rng: np.random.Generator
) -> None:
    """A ratio no larger than a spectrum's spacing without a kernel is not trusted.

    That spacing reaches at most 19.7 over the ladder's builds, so such a ratio could be spacing.
    """
    state = reference_state(_with_spectrum([4.0, 3.0, 2.0, 1.0, 1.0 / ratio], rng))
    assert state.kernel_dimension == 1
    assert state.gap_ratio == pytest.approx(ratio)
    assert state.trusted is trusted


@pytest.mark.gotcha
def test_fig3b_at_n3_clears_the_threshold_and_r1_at_n2_does_not() -> None:
    """Fig. 3b's operator at n = 3 has a gap ratio of 66; R1's at n = 2 has 10.9."""
    n = 3
    state = reference_state(_standard_residual(n, (1.0, 4.0, 4.0), zero_row(-1.0, n)))
    assert state.gap_ratio == pytest.approx(69.4, rel=0.05)
    assert state.trusted
    coarse = physical_reference(assemble(problem("R1"), 2))
    assert coarse.kernel_dimension == 1
    assert coarse.gap_ratio < classical.DEFAULT_MIN_GAP_RATIO
    assert not coarse.trusted


@pytest.mark.gotcha
@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int64])
def test_lower_precision_is_refused(dtype: Any) -> None:
    """Float32 contamination once capped accuracy at 8.65e-8; it is refused outright."""
    with pytest.raises(TypeError, match="single precision"):
        reference_state(np.eye(3, dtype=dtype))


def test_sparse_input_is_accepted() -> None:
    R = sp.csr_array(np.diag([3.0, 2.0, 1e-12]))
    assert reference_state(R).kernel_dimension == 1


@pytest.mark.gotcha
def test_gesdd_failure_falls_back_to_gesvd(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    """When gesdd does not converge, gesvd is used and the answer is unchanged."""
    R = _with_spectrum([4.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 1e-13], rng)
    expected = reference_state(R)
    real_svd = scipy.linalg.svd

    def failing_gesdd(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("lapack_driver") == "gesdd":
            raise np.linalg.LinAlgError("SVD did not converge")
        return real_svd(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "svd", failing_gesdd)
    state = reference_state(R)
    assert state.driver == "gesvd"
    assert expected.driver == "gesdd"
    np.testing.assert_allclose(state.vector, expected.vector, atol=1e-12)
    # The 1e-13 value sits at the noise floor: the drivers agree to eps * sigma_max, not to a
    # relative tolerance (earlier code failed on exactly that).
    np.testing.assert_allclose(
        state.singular_values, expected.singular_values, rtol=1e-10, atol=1e-13
    )
    assert state.kernel_dimension == expected.kernel_dimension == 1
    assert state.trusted
    assert expected.trusted


@pytest.mark.slow
@pytest.mark.gotcha
def test_recorded_operator_gives_the_same_answer_with_either_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded matrix: the Carleman-NS system of the earlier failure (6944 x 3472).

    gesdd did not converge on it where it was recorded; it does converge here,
    so both drivers are run and must agree on a trusted one-dimensional kernel.
    """
    with np.load(RECORDED) as data:
        R = sp.coo_array((data["data"], (data["row"], data["col"])), shape=tuple(data["shape"]))
    dense = R.toarray()
    by_gesdd = reference_state(dense)
    real_svd = scipy.linalg.svd

    def failing_gesdd(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("lapack_driver") == "gesdd":
            raise np.linalg.LinAlgError("SVD did not converge")
        return real_svd(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "svd", failing_gesdd)
    by_gesvd = reference_state(dense)
    assert by_gesvd.driver == "gesvd"
    assert by_gesdd.kernel_dimension == by_gesvd.kernel_dimension == 1
    assert by_gesdd.trusted
    assert by_gesvd.trusted
    assert abs(np.vdot(by_gesdd.vector, by_gesvd.vector)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.gotcha
@pytest.mark.parametrize("n", range(4, 9))
def test_svd_reference_is_exact_where_eigh_fails(n: int) -> None:
    """SVD of R stays exact; eigh(R^T R) degrades from n = 6.

    Fig. 3b's equation with the invariant f(-1/2) = 0 has the solution (1/2 + x) e^{-2x}.
    Measured 1 - |cos| against its normalised coefficients: SVD <= 2e-15 for n = 4..8;
    eigh 5e-12, 2e-6 and 0.72 at n = 6, 7 and 8.
    """
    R = _standard_residual(n, (1.0, 4.0, 4.0), zero_row(-0.5, n))
    target = basis.fit(lambda x: (0.5 + x) * np.exp(-2 * x), n)
    target /= np.linalg.norm(target)
    state = reference_state(R)
    assert state.trusted
    assert 1 - abs(state.vector @ target) < 1e-13
    if n == 8:
        _, vectors = np.linalg.eigh(R.T @ R)
        assert 1 - abs(vectors[:, 0] @ target) > 1e-2


@pytest.mark.parametrize("name", ["R2a", "R3c", "R5b"])
def test_readout_bound_holds_for_states_near_and_far_from_the_reference(name: str) -> None:
    """The decoded field of states around the ground state stays within eps_disc plus the bound.

    Random states at distances 1e-8 to 1e-1, each with a random global phase, wherever the bound
    applies (:class:`pihm.classical.ReadoutBound`).
    """
    p = problem(name, "corrected")
    residual = assemble(p, (3,) * len(p.axes))
    solution = solve(residual)
    reference = np.asarray(solution.vector, dtype=np.complex128)
    terms = readout_bound(residual, reference)
    rng = np.random.default_rng(7)
    checked = 0
    for size in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        for _ in range(4):
            noise = rng.normal(size=reference.size) + 1j * rng.normal(size=reference.size)
            state = reference + size * noise / np.linalg.norm(noise)
            state *= np.exp(1j * rng.uniform(0, 2 * np.pi)) / np.linalg.norm(state)
            overlap = np.vdot(reference, state)
            distance = float(np.linalg.norm(state * np.exp(-1j * np.angle(overlap)) - reference))
            extra = terms.bound(distance)
            if not np.isfinite(extra):
                continue
            linf, _ = field_error(residual, state)
            assert linf <= solution.field_error_linf + extra
            checked += 1
    assert checked >= 12


def test_the_physical_reference_leaves_the_padding_at_zero() -> None:
    """The padding decouples: its spectrum is left out, its coefficients zero."""
    residual = assemble(problem("R2a", "corrected"), 3, regime="descriptor")
    state = physical_reference(residual)
    size = residual.dimension // len(residual.derivatives)
    assert state.vector.shape == (residual.dimension,)
    assert state.singular_values.size == residual.physical_columns.size
    for i, derivative in enumerate(residual.derivatives):
        if derivative is None:
            assert np.all(state.vector[i * size : (i + 1) * size] == 0.0)
    standard = assemble(problem("R2a", "corrected"), 3)
    whole = reference_state(standard.matrix())
    np.testing.assert_array_equal(physical_reference(standard).vector, whole.vector)


@pytest.mark.parametrize("regime", ["standard", "descriptor"])
@pytest.mark.parametrize(("name", "order", "n"), [("R8", 3, 5), ("R9c-M4", 3, 4)])
def test_a_carleman_build_splits_its_error_between_the_lift_and_the_time_axis(
    name: str, order: int, n: int, regime: str
) -> None:
    """The lift's floor carries the field's error; the resolved time axis adds almost nothing."""
    solution = solve(assemble(problem(name), n, regime=regime, carleman=order))  # type: ignore[arg-type]
    lift, axis = lift_errors(solution.residual, solution.vector)
    assert abs(solution.field_error_linf - lift) <= axis + 1e-12
    assert axis < 1e-6 * lift


def test_only_a_carleman_lift_has_an_error_split() -> None:
    solution = solve(assemble(problem("R7"), 3))
    with pytest.raises(ValueError, match="Carleman"):
        lift_errors(solution.residual, solution.vector)


@pytest.mark.parametrize(
    ("name", "n", "order", "regime"),
    [
        ("R2a", 8, None, "descriptor"),
        ("R2c", 7, None, "descriptor"),
        ("R7", 7, None, "standard"),
        ("R9c-M4", 4, 3, "descriptor"),
        ("R10a-M12", 3, 2, "standard"),
        ("R10a-M12", 3, 2, "descriptor"),
        ("R10c-M12", 3, 2, "descriptor"),
    ],
)
def test_the_iterative_reference_agrees_with_the_dense_svd(
    name: str, n: int, order: int | None, regime: str
) -> None:
    """The smallest spectrum to a few eps sigma_max, sigma_max and the state to rounding.

    R10's spectra are degenerate by symmetry (Taylor-Green's lowest 17 hold a value ten times), so
    they check that the start vector meets every symmetry sector.
    """
    residual = assemble(problem(name), n, regime=regime, carleman=order)  # type: ignore[arg-type]
    R = sp.csr_array(residual.matrix()[:, residual.physical_columns])
    dense, iterative = reference_state(R), iterative_reference_state(R)
    sigma_max = float(dense.singular_values[-1])
    assert iterative.driver == "iterative"
    assert not iterative.complete
    assert np.max(np.abs(iterative.singular_values[:17] - dense.singular_values[:17])) < (
        10 * np.finfo(float).eps * sigma_max
    )
    assert iterative.singular_values[-1] == pytest.approx(sigma_max, rel=1e-12)
    assert 1.0 - abs(np.vdot(iterative.vector, dense.vector)) ** 2 < 1e-12
    assert iterative.backward_error < 1e-14
    assert iterative.trusted == dense.trusted
    assert iterative.fidelity_bound is not None
    s0, s1 = dense.singular_values[:2]
    assert iterative.fidelity_bound <= 1.001 * s0**2 / (s1**2 - s0**2) + 1e-12


def test_past_the_dense_ceiling_the_reference_is_iterative(monkeypatch: pytest.MonkeyPatch) -> None:
    residual = assemble(problem("R7"), 6, regime="descriptor")
    dense = physical_reference(residual)
    monkeypatch.setattr(classical, "DENSE_COLUMNS", 64)
    iterative = physical_reference(residual)
    assert (dense.driver, iterative.driver) == ("gesdd", "iterative")
    assert 1.0 - abs(np.vdot(iterative.vector, dense.vector)) ** 2 < 1e-12
    assert iterative.gap == pytest.approx(dense.gap, rel=1e-10)


def test_the_iterative_reference_refuses_too_few_columns_and_single_precision() -> None:
    with pytest.raises(ValueError, match="more than 17 columns"):
        iterative_reference_state(np.eye(10))
    with pytest.raises(TypeError, match="float64"):
        iterative_reference_state(np.eye(40, dtype=np.float32))


def test_a_matrix_past_32_bit_lapacks_workspace_leaves_it() -> None:
    """The gesdd workspace query is right up to the bound, which R10's K = 4 passes."""
    from pihm.classical.reference import gesdd_fits

    largest = 20_723
    assert gesdd_fits((largest, largest))
    assert not gesdd_fits((largest + 1, largest + 1))
    assert not gesdd_fits((34815, 29104))
    for m, n in ((largest, largest), (2 * largest, largest), (34815, 19584)):
        work, info = scipy.linalg.lapack.dgesdd_lwork(m, n, compute_uv=1, full_matrices=0)
        assert info == 0
        assert work >= 3 * n * n + 4 * n
