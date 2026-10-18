"""Fourier–Galerkin fluids: Burgers and Cole–Hopf (R9), Navier–Stokes and its DNS (R10)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.assemble.systems import vector_field
from pihm.carleman import nonlinearity_ratio
from pihm.problems import ladder
from pihm.problems.fluids import (
    Galerkin,
    Mode,
    burgers,
    burgers_couplings,
    burgers_galerkin,
    cole_hopf,
    fourier_field,
    mode_name,
    real_embedding,
    sine_modes,
    taylor_green,
    vortex_pair,
    vorticity_couplings,
    vorticity_dns,
    vorticity_field,
    vorticity_galerkin,
    vorticity_modes,
)
from pihm.problems.reference import check_reference, reference_solution, slice_data
from pihm.problems.spec import Problem


def test_the_couplings_are_the_galerkin_projection_of_the_nonlinear_term() -> None:
    """-(u^2/2)_x projected on sin kx, by quadrature, against the closed-form couplings."""
    modes = 5
    rng = np.random.default_rng(11)
    b = rng.normal(size=modes)
    x = 2 * np.pi * np.arange(256) / 256
    k = np.arange(1, modes + 1)
    u = np.sin(np.outer(x, k)) @ b
    u_x = np.cos(np.outer(x, k)) @ (k * b)
    projected = 2.0 * (-(u * u_x)) @ np.sin(np.outer(x, k)) / x.size
    closed = np.zeros(modes)
    for (target, j, m), value in burgers_couplings(modes).items():
        closed[target - 1] += value * b[j - 1] * b[m - 1]
    np.testing.assert_allclose(closed, projected, atol=1e-12)


def test_cole_hopf_solves_burgers() -> None:
    a, nu = 0.8, 0.5
    u = cole_hopf(a, nu)
    x, s, h = np.linspace(0.1, 6.0, 40), 0.37, 1e-4
    u_s = (u(x, s + h) - u(x, s - h)) / (2 * h)
    u_x = (u(x + h, s) - u(x - h, s)) / (2 * h)
    u_xx = (u(x + h, s) - 2 * u(x, s) + u(x - h, s)) / h**2
    assert np.max(np.abs(u_s + u(x, s) * u_x - nu * u_xx)) < 1e-6
    np.testing.assert_allclose(u(x, 0.0), a * np.sin(x), atol=1e-13)


@pytest.mark.parametrize("modes", [4, 8])
def test_the_galerkin_reference_converges_to_cole_hopf(modes: int) -> None:
    problem = burgers(modes, 1.0, 1.0, 1.0)
    check = check_reference(problem)
    assert check.equation < 1e-8
    t = np.linspace(-1.0, 1.0, 11)
    galerkin = np.array(
        [reference_solution(problem, f"b{k}").field(t) for k in range(1, modes + 1)]
    )
    exact = sine_modes(cole_hopf(1.0, 1.0), modes, (t + 1) / 2).T
    assert np.max(np.abs(galerkin - exact)) < (1e-5 if modes == 4 else 1e-9)


def test_the_ratio_is_taken_on_symmetric_tensors() -> None:
    """R = |u0| |F_2 sym| / nu: the same for any order of the products' factors."""
    field = vector_field(burgers(4, 1.0, 1.0, 1.0))
    swapped = field[1].toarray().reshape(4, 4, 4).transpose(0, 2, 1).reshape(4, 16)
    same = nonlinearity_ratio([field[0], sp.csr_array(swapped)], [1.0, 0, 0, 0])
    assert same == pytest.approx(nonlinearity_ratio(field, [1.0, 0, 0, 0]), rel=1e-14)


def test_the_ladder_names_r9_by_its_strength_and_modes() -> None:
    """Five strengths a/nu on 4, 8 and 16 modes, nu = T = 1."""
    assert len(ladder.names("R9")) == len(ladder.R9_STRENGTHS) * len(ladder.R9_MODES)
    assert ladder.burgers_panel("R9c-M8") == (1.0, 8)
    assert ladder.problem("R9c-M8").digest == burgers(8, 1.0, 1.0, 1.0, name="R9c-M8").digest
    with pytest.raises(ValueError, match="not an R9 problem"):
        ladder.burgers_panel("R8")


def test_the_reference_holds_the_stiff_modes_at_sixteen_modes() -> None:
    """Radau keeps M = 16's fastest modes at the tolerance, where an explicit method did not."""
    problem = ladder.problem("R9a-M16")
    t = np.linspace(-1.0, 1.0, 21)
    galerkin = np.array([reference_solution(problem, f"b{k}").field(t) for k in range(1, 17)])
    exact = sine_modes(cole_hopf(0.03, 1.0), 16, (t + 1) / 2).T
    assert np.max(np.abs(galerkin - exact)) < 1e-12


# ---------------------------------------------------------------------------- R10


def _gradient(
    modes: tuple[Mode, ...], w: np.ndarray, x: np.ndarray, y: np.ndarray, power: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient of sum w_m phi_m / |k_m|^power on the grid, analytically."""
    gx, gy = np.zeros_like(x), np.zeros_like(x)
    for (kind, kx, ky), value in zip(modes, w, strict=True):
        scale = value / (kx * kx + ky * ky) ** power
        phase = kx * x + ky * y
        slope = -np.sin(phase) if kind == "cos" else np.cos(phase)
        gx += scale * kx * slope
        gy += scale * ky * slope
    return gx, gy


def _quadratic(cutoff: int, w: np.ndarray) -> np.ndarray:
    """Return sum c_kab w_a w_b from the exact couplings."""
    result = np.zeros(len(w))
    for (k, a, b), value in vorticity_couplings(cutoff).items():
        result[k] += float(value) * w[a] * w[b]
    return result


def test_the_modes_are_the_half_plane_by_wavenumber() -> None:
    assert [len(vorticity_modes(c)) for c in (1, 2, 4, 5, 8, 9)] == [4, 8, 12, 20, 24, 28]
    assert vorticity_modes(2)[:4] == (("cos", 0, 1), ("sin", 0, 1), ("cos", 1, 0), ("sin", 1, 0))
    assert mode_name(("sin", 1, -2)) == "s_1_m2"


@pytest.mark.parametrize("cutoff", [2, 5, 9])
def test_the_couplings_are_the_galerkin_projection_of_the_jacobian(cutoff: int) -> None:
    """-J(psi, omega) on a grid, projected on each mode by quadrature, against the couplings."""
    modes = vorticity_modes(cutoff)
    w = np.random.default_rng(cutoff).normal(size=len(modes))
    grid = 2 * np.pi * np.arange(32) / 32
    x, y = np.meshgrid(grid, grid, indexing="ij")
    psi_x, psi_y = _gradient(modes, w, x, y, 1)
    omega_x, omega_y = _gradient(modes, w, x, y, 0)
    minus_j = -(psi_x * omega_y - psi_y * omega_x)
    ones = np.eye(len(modes))
    projected = [2 * np.mean(minus_j * vorticity_field(modes, e, x, y)) for e in ones]
    np.testing.assert_allclose(_quadratic(cutoff, w), projected, atol=1e-12)


@pytest.mark.parametrize("cutoff", [4, 9])
def test_the_nonlinear_term_conserves_energy_and_enstrophy_exactly(cutoff: int) -> None:
    """The truncated Jacobian keeps sum w^2/|k|^2 and sum w^2, in exact arithmetic."""
    modes = vorticity_modes(cutoff)
    rng = np.random.default_rng(1)
    w = [Fraction(int(v)) for v in rng.integers(-9, 10, size=len(modes))]
    slope = [Fraction(0)] * len(modes)
    for (k, a, b), value in vorticity_couplings(cutoff).items():
        slope[k] += value * w[a] * w[b]
    sizes = [kx * kx + ky * ky for _, kx, ky in modes]
    assert sum(wk * sk / size for wk, sk, size in zip(w, slope, sizes, strict=True)) == 0
    assert sum(wk * sk for wk, sk in zip(w, slope, strict=True)) == 0


def test_taylor_green_is_an_exact_decay_in_the_truncation_and_the_equation() -> None:
    """Its modes share |k|, so the Jacobian vanishes: the reference, and the DNS, decay exactly."""
    problem = taylor_green(4, 0.5, 1.0)
    modes = vorticity_modes(4)
    w0 = slice_data(problem, "t", -1.0)
    assert not _quadratic(4, w0).any()
    check = check_reference(problem)
    assert check.equation < 1e-13
    assert check.data == 0.0
    s = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(
        vorticity_dns(4, w0, 0.5, s), np.exp(-2 * 0.5 * s)[:, None] * w0, atol=1e-10
    )
    grid = np.linspace(0.0, 2 * np.pi, 9)
    np.testing.assert_allclose(
        vorticity_field(modes, w0, grid, grid[::-1]),
        2 * np.sin(grid) * np.sin(grid[::-1]),
        atol=1e-14,
    )


def test_the_vortex_pair_is_the_exact_projection_of_two_gaussians() -> None:
    """The closed-form data against a quadrature of the periodised pair, scaled to norm a."""
    problem = vortex_pair(9, 3.0, 1.0, 1.0)
    modes = vorticity_modes(9)
    data = slice_data(problem, "t", -1.0)
    grid = 2 * np.pi * np.arange(64) / 64
    x, y = np.meshgrid(grid, grid, indexing="ij")
    pair = np.zeros_like(x)
    for sign, cx in ((1.0, 0.75 * np.pi), (-1.0, 1.25 * np.pi)):
        for dx in (-2 * np.pi, 0.0, 2 * np.pi):
            for dy in (-2 * np.pi, 0.0, 2 * np.pi):
                pair += sign * np.exp(-((x - cx - dx) ** 2 + (y - np.pi - dy) ** 2) / 0.5)
    projected = np.array([2 * np.mean(pair * vorticity_field(modes, e, x, y)) for e in np.eye(28)])
    np.testing.assert_allclose(data, 3.0 * projected / np.linalg.norm(projected), atol=1e-12)
    assert np.linalg.norm(data) == pytest.approx(3.0, rel=1e-14)
    assert problem.constraints[0].field == "s_1_0"


def test_the_dns_slope_is_the_galerkin_slope_inside_the_truncation() -> None:
    """Over a short step from data inside the truncation, the equation moves as its truncation."""
    modes = vorticity_modes(5)
    w0 = np.random.default_rng(5).normal(size=len(modes))
    h = 1e-5
    step = vorticity_dns(5, w0, 1.0, [0.0, h], tolerance=1e-13)
    sizes = np.array([kx * kx + ky * ky for _, kx, ky in modes])
    slope = -sizes * w0 + _quadratic(5, w0)
    np.testing.assert_allclose((step[1] - step[0]) / h, slope, rtol=0, atol=1e-3)


def test_the_ladder_names_r10_by_its_panel_modes_and_window() -> None:
    """Every panel on 12-48 modes, two on 80, the pair at 10 at field scale."""
    assert len(ladder.names("R10")) == 4 * 4 + 4 + 2 + 2
    assert ladder.navier_stokes_panel("R10c-M20") == (10.0, 5, 1.0)
    assert ladder.navier_stokes_panel("R10d-M12-W8") == (30.0, 4, 0.125)
    assert ladder.navier_stokes_panel("R10a-M28") == (None, 9, 1.0)
    assert [ladder.rung(name) for name in ("R10c-M20", "R2a", "R9")] == ["R10", "R2", "R9"]
    assert (
        ladder.problem("R10b-M12").digest == vortex_pair(4, 3.0, 1.0, 1.0, name="R10b-M12").digest
    )
    for bad in ("R10e-M12", "R10a-M16", "R10c-M12-W4", "R9c-M8"):
        with pytest.raises(ValueError, match="not an R10 problem"):
            ladder.navier_stokes_panel(bad)


# ---------------------------------------------------------------------------- the structured form


@pytest.mark.parametrize(
    ("galerkin", "problem"),
    [
        (burgers_galerkin(4, 1.0, 1.0), burgers(4, 1.0, 1.0, 1.0)),
        (burgers_galerkin(8, 0.7, 0.5), burgers(8, 2.0, 0.7, 0.5)),
        (vorticity_galerkin(4, 1.0, 1.0), vortex_pair(4, 3.0, 1.0, 1.0)),
        (vorticity_galerkin(9, 1.0, 0.125), vortex_pair(9, 3.0, 1.0, 0.125)),
    ],
    ids=["R9-M4", "R9-M8", "R10-M12", "R10-M28-W8"],
)
def test_the_structured_form_is_the_problem_through_the_real_embedding(
    galerkin: Galerkin, problem: Problem
) -> None:
    """E^dagger F^v(E w, ..., E w) = F(w, ..., w) for the problem's own vector field."""
    F1, F2 = fourier_field(galerkin)
    E = real_embedding(galerkin).toarray()
    R1, R2 = vector_field(problem)
    np.testing.assert_allclose(E.conj().T @ E, np.eye(E.shape[1]), atol=1e-15)
    w = np.random.default_rng(2).normal(size=E.shape[1])
    v = E @ w
    np.testing.assert_allclose(E.conj().T @ (F1 @ v), R1 @ w, atol=1e-13)
    np.testing.assert_allclose(E.conj().T @ (F2 @ np.kron(v, v)), R2 @ np.kron(w, w), atol=1e-13)


def test_the_vorticity_convolution_is_real_and_burgers_imaginary() -> None:
    """Navier-Stokes' two i's cancel; Burgers keeps its derivative's."""
    assert not fourier_field(vorticity_galerkin(5, 1.0, 1.0))[1].toarray().imag.any()
    assert not fourier_field(burgers_galerkin(4, 1.0, 1.0))[1].toarray().real.any()
