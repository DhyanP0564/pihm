"""A Fourier–Galerkin problem's fields on its periodic domain (``pihm.problems.periodic``)."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.problems.fluids import mode_name, sine_modes
from pihm.problems.ladder import problem
from pihm.problems.periodic import PeriodicSpace, periodic_space

#: A uniform grid on the periodic square, on which the trapezoidal rule is exact for the
#: truncations' trigonometric polynomials.
POINTS = 64
AXIS = 2.0 * np.pi * np.arange(POINTS) / POINTS
X, Y = np.meshgrid(AXIS, AXIS, indexing="ij")


@pytest.mark.parametrize("name", ["R10a-M12", "R10c-M20", "R10d-M28-W8"])
def test_the_vorticity_modes_are_the_problems_fields_in_order(name: str) -> None:
    space = periodic_space(name)
    assert space is not None
    assert [mode_name(mode) for mode in space.modes] == [f.name for f in problem(name).fields]


@pytest.mark.parametrize("name", ["R9a-M4", "R9c-M8", "R9e-M16"])
def test_burgers_sine_modes_are_the_problems_fields_in_order(name: str) -> None:
    space = periodic_space(name)
    assert space is not None
    assert [f"b{kx}" for _, kx, _ in space.modes] == [f.name for f in problem(name).fields]
    assert {ky for _, _, ky in space.modes} == {0}


def test_only_the_fourier_galerkin_rungs_have_a_periodic_space() -> None:
    assert periodic_space("R2c") is None
    assert periodic_space("R8") is None
    window = periodic_space("R10d-M20-W8")
    assert window is not None
    assert window.time == 0.125


def test_the_taylor_green_vortex_expands_to_its_closed_form() -> None:
    space = periodic_space("R10a-M12")
    assert space is not None
    start = np.zeros(len(space.modes))
    start[space.modes.index(("cos", 1, -1))] = 1.0
    start[space.modes.index(("cos", 1, 1))] = -1.0
    decay = np.exp(-2.0 * space.seconds(np.array([-1.0, 0.0, 1.0])))
    omega = space.field(np.outer(start, decay), X, Y)
    exact = 2.0 * np.sin(X)[..., None] * np.sin(Y)[..., None] * decay
    assert omega.shape == (POINTS, POINTS, 3)
    assert np.max(np.abs(omega - exact)) < 1e-14


def test_the_energy_and_enstrophy_are_the_domain_averages() -> None:
    """E = ½⟨|u|²⟩ = ½⟨ψω⟩ by parts on the torus, and Z = ½⟨ω²⟩, by quadrature."""
    space = periodic_space("R10c-M20")
    assert space is not None
    amplitudes = np.random.default_rng(7).normal(size=len(space.modes))
    omega = space.field(amplitudes, X, Y)
    psi = space.stream(amplitudes, X, Y)
    assert float(space.energy(amplitudes)) == pytest.approx(0.5 * np.mean(psi * omega), rel=1e-12)
    assert float(space.enstrophy(amplitudes)) == pytest.approx(0.5 * np.mean(omega**2), rel=1e-12)


def test_the_stream_function_inverts_the_laplacian() -> None:
    space = periodic_space("R10b-M12")
    assert space is not None
    amplitudes = np.random.default_rng(3).normal(size=(len(space.modes), 2))
    step = AXIS[1]
    psi = space.stream(amplitudes, X, Y)
    spectrum = np.fft.fft2(psi, axes=(0, 1))
    k = np.fft.fftfreq(POINTS, step / (2.0 * np.pi))
    laplacian = -(k[:, None] ** 2 + k[None, :] ** 2)
    minus_omega = np.fft.ifft2(laplacian[..., None] * spectrum, axes=(0, 1)).real
    assert np.max(np.abs(minus_omega + space.field(amplitudes, X, Y))) < 1e-11  # FFT rounding


def test_burgers_modes_expand_to_the_cole_hopf_solution() -> None:
    """Sine coefficients of the exact solution, expanded again, return it (a = ν = 1, M = 16)."""
    space = periodic_space("R9c-M16")
    assert space is not None
    s = np.array([0.0, 0.3, 1.0])
    coefficients = sine_modes(lambda x, t: space.exact(x, t), len(space.modes), s).T
    u = space.field(coefficients, AXIS)
    assert u.shape == (POINTS, s.size)
    assert np.max(np.abs(u - space.exact(AXIS[:, None], s[None, :]))) < 1e-12
    assert float(space.energy(coefficients)[0]) == pytest.approx(0.25, rel=1e-12)


def test_burgers_has_neither_a_stream_function_nor_an_enstrophy() -> None:
    space = PeriodicSpace("burgers", (("sin", 1, 0),), viscosity=1.0, time=1.0, amplitude=1.0)
    with pytest.raises(ValueError, match="stream function"):
        space.stream(np.ones(1), AXIS, AXIS)
    with pytest.raises(ValueError, match="enstrophy"):
        space.enstrophy(np.ones(1))
    vorticity = periodic_space("R10a-M12")
    assert vorticity is not None
    with pytest.raises(ValueError, match="closed form"):
        vorticity.exact(AXIS, 0.0)
