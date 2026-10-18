r"""A Fourier–Galerkin problem's fields on its periodic domain (R9, R10).

A Fourier–Galerkin problem (:mod:`pihm.problems.fluids`) carries one field per Fourier mode it
keeps, each a function of the Chebyshev time :math:`t \in [-1, 1]`, where :math:`s = T(t + 1)/2`.
:class:`PeriodicSpace` expands those amplitudes on the domain again:

* viscous Burgers (R9): :math:`u(x, s) = \sum_k b_k(s)\sin kx` on :math:`[0, 2\pi)`;
* the vorticity (R10): :math:`\omega(x, y, s) = \sum_m w_m(s)\varphi_m(x, y)` on
  :math:`[0, 2\pi)^2`, with the stream function :math:`\psi`, :math:`\hat\psi_k =
  \hat\omega_k/|k|^2`, whose contours are the streamlines.

:func:`periodic_space` gives a ladder problem's own space from its name, as
:func:`pihm.problems.ladder.galerkin` gives its structured form, so a problem's identity holds
none of it.

**The averages over the domain** by which a flow's decay is read: every basis function has mean
square :math:`\tfrac12` on the domain, and distinct ones are orthogonal there, their gradients
too. Burgers' kinetic energy is therefore :math:`E = \tfrac12\langle u^2\rangle = \tfrac14\sum_k
b_k^2`. In two dimensions :math:`|\mathbf{u}|^2 = |\nabla\psi|^2`, so :math:`E = \tfrac14\sum_m
w_m^2/|k_m|^2`, and the enstrophy is :math:`Z = \tfrac12\langle\omega^2\rangle = \tfrac14\sum_m
w_m^2`.

Examples
--------
>>> import numpy as np
>>> space = PeriodicSpace("burgers", (("sin", 1, 0), ("sin", 2, 0)), viscosity=1.0, time=1.0)
>>> x = np.array([np.pi / 2])
>>> space.field(np.array([1.0, 0.5]), x)
array([1.])
>>> float(space.energy(np.array([1.0, 0.5])))
0.3125
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pihm.problems.fluids import Mode, cole_hopf, vorticity_modes
from pihm.problems.ladder import burgers_panel, navier_stokes_panel, rung

__all__ = ["Equation", "PeriodicSpace", "periodic_space"]

#: The equation a Fourier–Galerkin problem truncates, Burgers (R9) or the vorticity form of
#: two-dimensional Navier–Stokes (R10).
Equation: TypeAlias = Literal["burgers", "vorticity"]


@dataclass(frozen=True)
class PeriodicSpace:
    r"""The periodic domain a Fourier–Galerkin problem's fields expand a field on.

    Attributes
    ----------
    equation
        The equation truncated.
    modes
        The basis function of each field, in the problem's field order, as
        :data:`~pihm.problems.fluids.Mode`: :math:`\cos` or :math:`\sin` of
        :math:`k_x x + k_y y`. Burgers' sine modes are ``("sin", k, 0)``.
    viscosity
        :math:`\nu`.
    time
        :math:`T`: the Chebyshev time :math:`t` is :math:`s = T(t + 1)/2`.
    amplitude
        Burgers' initial amplitude :math:`a`, for its exact solution (:meth:`exact`); ``None``
        for the vorticity.
    """

    equation: Equation
    modes: tuple[Mode, ...]
    viscosity: float
    time: float
    amplitude: float | None = None

    @property
    def dimension(self) -> int:
        """The domain's number of axes, 1 for Burgers and 2 for the vorticity."""
        return 1 if self.equation == "burgers" else 2

    @property
    def wavenumbers(self) -> NDArray[np.float64]:
        r""":math:`|k_m|^2` of each mode, in field order."""
        return np.array([kx * kx + ky * ky for _, kx, ky in self.modes], dtype=np.float64)

    def seconds(self, t: ArrayLike) -> NDArray[np.float64]:
        r"""Return the time :math:`s = T(t + 1)/2` of the Chebyshev coordinate ``t``."""
        return np.asarray(0.5 * self.time * (np.asarray(t, np.float64) + 1.0))

    def basis(self, x: ArrayLike, y: ArrayLike = 0.0) -> NDArray[np.float64]:
        """Return every mode's basis function at the points, modes along a first axis.

        ``x`` and ``y`` broadcast; ``y`` is ignored on Burgers' interval, whose modes have
        :math:`k_y = 0`.
        """
        x_, y_ = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
        values = np.empty((len(self.modes), *x_.shape))
        for index, (kind, kx, ky) in enumerate(self.modes):
            phase = kx * x_ + ky * y_
            values[index] = np.cos(phase) if kind == "cos" else np.sin(phase)
        return values

    def field(
        self, coefficients: ArrayLike, x: ArrayLike, y: ArrayLike = 0.0
    ) -> NDArray[np.float64]:
        r"""Return :math:`u` or :math:`\omega` from its modes' amplitudes, at the points.

        Parameters
        ----------
        coefficients
            The amplitudes, modes along the first axis; further axes (the times, say) are kept.
        x, y
            The points, broadcasting against each other.

        Returns
        -------
        numpy.ndarray
            The points' shape, then the amplitudes' further axes.
        """
        return np.tensordot(self.basis(x, y), np.asarray(coefficients, np.float64), axes=(0, 0))

    def stream(self, coefficients: ArrayLike, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        r"""Return the stream function :math:`\psi`, :math:`\hat\psi_k = \hat\omega_k/|k|^2`.

        As :meth:`field`, from the vorticity's amplitudes.

        Raises
        ------
        ValueError
            For Burgers, which has no stream function.
        """
        if self.equation != "vorticity":
            raise ValueError("only the vorticity has a stream function")
        amplitudes = np.asarray(coefficients, np.float64)
        scale = self.wavenumbers.reshape((-1,) + (1,) * (amplitudes.ndim - 1))
        return np.tensordot(self.basis(x, y), amplitudes / scale, axes=(0, 0))

    def energy(self, coefficients: ArrayLike) -> NDArray[np.float64]:
        r"""Return the kinetic energy :math:`E = \tfrac12\langle|\mathbf{u}|^2\rangle`.

        Over the amplitudes' further axes (the times, say); the module documentation derives it.
        """
        squares = np.asarray(coefficients, np.float64) ** 2
        if self.equation == "vorticity":
            squares = squares / self.wavenumbers.reshape((-1,) + (1,) * (squares.ndim - 1))
        return np.asarray(0.25 * np.sum(squares, axis=0))

    def enstrophy(self, coefficients: ArrayLike) -> NDArray[np.float64]:
        r"""Return the enstrophy :math:`Z = \tfrac12\langle\omega^2\rangle` (module docs).

        Raises
        ------
        ValueError
            For Burgers, whose field is not a vorticity.
        """
        if self.equation != "vorticity":
            raise ValueError("the enstrophy is the vorticity's")
        return np.asarray(0.25 * np.sum(np.asarray(coefficients, np.float64) ** 2, axis=0))

    def exact(self, x: ArrayLike, s: ArrayLike) -> NDArray[np.float64]:
        r"""Return Burgers' exact solution :math:`u(x, s)`, which no truncation limits.

        The Cole–Hopf series (:func:`pihm.problems.fluids.cole_hopf`); ``x`` and ``s``
        broadcast. Against it the Galerkin reference's own error, the truncation's, shows.

        Raises
        ------
        ValueError
            For the vorticity, whose equation has no closed form here.
        """
        if self.equation != "burgers" or self.amplitude is None:
            raise ValueError("only Burgers has an exact solution in closed form")
        return cole_hopf(self.amplitude, self.viscosity)(x, s)


def periodic_space(name: str) -> PeriodicSpace | None:
    r"""Return an R9 or R10 problem's periodic space, from its name; ``None`` for any other.

    Both rungs have :math:`\nu = 1`; R9 runs over :math:`T = 1`, from :math:`a \sin x` with
    :math:`a = a/\nu` (:func:`~pihm.problems.ladder.burgers_panel`), and R10 over its panel's
    window (:func:`~pihm.problems.ladder.navier_stokes_panel`).

    Examples
    --------
    >>> space = periodic_space("R10c-M12")
    >>> space.equation, len(space.modes), space.time
    ('vorticity', 12, 1.0)
    >>> periodic_space("R2c") is None
    True
    """
    if rung(name) == "R9":
        strength, modes = burgers_panel(name)
        sines: tuple[Mode, ...] = tuple(("sin", k, 0) for k in range(1, modes + 1))
        return PeriodicSpace("burgers", sines, viscosity=1.0, time=1.0, amplitude=strength)
    if rung(name) == "R10":
        _, cutoff, time = navier_stokes_panel(name)
        return PeriodicSpace("vorticity", vorticity_modes(cutoff), viscosity=1.0, time=time)
    return None
