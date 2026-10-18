r"""Fourier–Galerkin fluids: Burgers (R9) and two-dimensional Navier–Stokes (R10).

Each is the Galerkin truncation of a periodic equation to a first-order polynomial system in
time, one field per Fourier mode kept, with its data as ratio rows at the start of the Chebyshev
time axis :math:`t = 2s/T - 1` (so every coefficient carries :math:`T/2`), and its reference the
system's own initial-value problem. Their Carleman lifts are rungs R9 and R10. The truncation's own
error, apart from the lift's and the time axis's, is
measured against an exact solution (:func:`cole_hopf`) or a pseudo-spectral simulation of the
equation itself (:func:`vorticity_dns`).

**Viscous Burgers** on the periodic interval :math:`x \in [0, 2\pi)`,

.. math::

   u_s + u u_x = \nu u_{xx}, \qquad u(x, 0) = a \sin x,

keeps an odd solution odd, so it lives on the sine modes, :math:`u = \sum_k b_k(s)\sin kx`. Its
Galerkin truncation to :math:`k \le M` (:func:`burgers`) is the quadratic first-order system

.. math::

   b_k' = -\nu k^2 b_k - \frac k4\Bigl[\sum_{j+l=k} b_j b_l - \sum_{|j-l|=k} b_j b_l\Bigr],

the coefficient of :math:`\sin kx` in :math:`-(u^2/2)_x + \nu u_{xx}`, with :math:`j, l \le M`.
Time :math:`s \in [0, T]` is carried on the Chebyshev axis :math:`t = 2s/T - 1`, so every
coefficient carries :math:`T/2`. Its data, :math:`b(0) = (a, 0, \dots, 0)`, enter as ratio rows,
and its reference is the system's own initial-value problem, integrated to the reference's
tolerance (the Galerkin truncation's "DNS"). The Carleman lift of it is rung R9's.

**The exact solution** (:func:`cole_hopf`): :math:`u = -2\nu\,\partial_x\ln\varphi`, with
:math:`\varphi` the heat equation's solution from :math:`\varphi(x, 0) =
\exp(-\beta(1 - \cos x))`, :math:`\beta = a/2\nu`. Expanded in the modified Bessel functions,
:math:`e^{\beta\cos x} = I_0(\beta) + 2\sum_n I_n(\beta)\cos nx`,

.. math::

   u(x, s) = \frac{4\nu\sum_n n\,I_n(\beta)\,e^{-\nu n^2 s}\sin nx}
                  {I_0(\beta) + 2\sum_n I_n(\beta)\,e^{-\nu n^2 s}\cos nx},

which measures the Galerkin truncation's own error, apart from the lift's (:func:`sine_modes`).

**Two-dimensional Navier–Stokes** in vorticity form on the periodic square
:math:`\mathbf{x} \in [0, 2\pi)^2`,

.. math::

   \omega_s + J(\psi, \omega) = \nu\nabla^2\omega, \qquad \nabla^2\psi = -\omega, \qquad
   J(\psi, \omega) = \psi_x\omega_y - \psi_y\omega_x,

with the stream function eliminated, :math:`\hat\psi_k = \hat\omega_k/|k|^2`. The vorticity is
carried in the real basis :math:`\cos(k\cdot\mathbf{x})`, :math:`\sin(k\cdot\mathbf{x})` over the
wavevectors of the half-plane :math:`k_x > 0` or :math:`k_x = 0 < k_y` with :math:`0 < |k|^2 \le
K_c` (:func:`vorticity_modes`); the mean vorticity is invariant and zero. With
:math:`\nabla\cos(p\cdot\mathbf{x}) = -p\sin(p\cdot\mathbf{x})` and :math:`\nabla\sin(p\cdot
\mathbf{x}) = p\cos(p\cdot\mathbf{x})`, the Jacobian of two basis functions is

.. math::

   J(\varphi_a/|p|^2, \varphi_b) = \sigma_a\sigma_b\,\frac{p \times q}{|p|^2}\,
   h_a(p\cdot\mathbf{x})\,h_b(q\cdot\mathbf{x}),

where :math:`p \times q = p_xq_y - p_yq_x`, :math:`\sigma = -1` and :math:`h = \sin` for a cosine,
:math:`\sigma = 1` and :math:`h = \cos` for a sine. The product of two such functions is a sum of
modes :math:`p \pm q`, of which the Galerkin truncation keeps those in the set, folded into the
half-plane (:func:`vorticity_couplings`). Every coefficient is rational, and is computed exactly.
Modes of one :math:`|k|` do not interact (the coefficients of :math:`(a, b)` and :math:`(b, a)`
cancel, as the complex form's :math:`(p \times q)(|p|^{-2} - |q|^{-2})` shows), so the
Taylor–Green vortex :math:`\omega = 2\sin x\sin y\,e^{-2\nu s}` is exact in any truncation that
holds :math:`|k|^2 = 2` (:func:`taylor_green`). A counter-rotating pair of Gaussian vortices,
projected exactly on the modes, interacts through its triads of unequal :math:`|k|`
(:func:`vortex_pair`).

**The structured form** (:class:`Galerkin`) states each equation by its Fourier symbols rather
than its couplings, :math:`u_s = L_1u + \sum \text{output}\bigl((\text{left}\,u)\cdot
(\text{right}\,u)\bigr)` with every operator a multiplier (:class:`Symbol`) and the product
pointwise: Burgers' :math:`-\tfrac12\partial_x(u^2)`, and Navier–Stokes' :math:`-(\psi_x\omega_y
- \psi_y\omega_x)` with :math:`\hat\psi = \hat\omega/|k|^2`. It is what a circuit builds
:math:`F_2 = L \circ \text{fold}` from. Its variables are
:math:`v = \sqrt2\,\hat u` on the truncation's wavevectors of both signs (:func:`fourier_field`),
so that the problem's real basis maps into them isometrically (:func:`real_embedding`), and the
two forms of the vector field agree through that map (``tests/unit/test_fluids.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special

from pihm.problems.expressions import compile_expression
from pihm.problems.spec import (
    Axis,
    Constraint,
    Datum,
    Equation,
    Factor,
    Field,
    Problem,
    Reference,
    Regular,
    Term,
)

__all__ = [
    "Galerkin",
    "Mode",
    "Product",
    "Symbol",
    "burgers",
    "burgers_couplings",
    "burgers_galerkin",
    "cole_hopf",
    "fourier_field",
    "mode_name",
    "navier_stokes",
    "real_embedding",
    "sine_modes",
    "taylor_green",
    "vortex_pair",
    "vorticity_couplings",
    "vorticity_dns",
    "vorticity_field",
    "vorticity_galerkin",
    "vorticity_modes",
]

#: Relative size below which a Cole–Hopf series term is dropped.
_SERIES_TOLERANCE = 1.0e-17


def burgers_couplings(modes: int) -> dict[tuple[int, int, int], float]:
    r"""Return the Galerkin coefficients :math:`c_{kjl}` of :math:`-(u^2/2)_x` on the sine modes.

    :math:`(-(u^2/2)_x)_k = \sum_{j \le l} c_{kjl} b_j b_l`, each unordered pair once, for
    :math:`j, l, k \le` ``modes``: :math:`-k/4` per ordered pair with :math:`j + l = k` and
    :math:`+k/4` per ordered pair with :math:`|j - l| = k`.
    """
    couplings: dict[tuple[int, int, int], float] = {}
    for k in range(1, modes + 1):
        for j in range(1, modes + 1):
            for m in range(j, modes + 1):
                ordered = 1 if j == m else 2
                value = 0.0
                if j + m == k:
                    value -= ordered * k / 4.0
                if abs(j - m) == k:
                    value += ordered * k / 4.0
                if value:
                    couplings[(k, j, m)] = value
    return couplings


def burgers(
    modes: int,
    amplitude: float,
    viscosity: float,
    time: float,
    *,
    name: str = "R9",
    tolerance: float = 1.0e-13,
) -> Problem:
    r"""Return the Galerkin truncation of viscous Burgers on ``modes`` sine modes (module docs).

    Parameters
    ----------
    modes
        :math:`M`, the highest sine mode kept.
    amplitude, viscosity
        :math:`a` and :math:`\nu` in :math:`u(x, 0) = a\sin x` and :math:`u_s + uu_x = \nu
        u_{xx}`.
    time
        :math:`T`, the end of the time interval.
    name
        The problem's name.
    tolerance
        The initial-value reference's tolerance.

    Returns
    -------
    Problem
        Fields ``b1`` to ``bM`` on the time axis ``t``, one equation each, the ratio constraint
        at :math:`t = -1`, the regular point there on ``b1``.
    """
    if modes < 1 or amplitude == 0 or viscosity <= 0 or time <= 0:
        raise ValueError("Burgers needs a mode, a non-zero amplitude, and positive nu and T")
    half = time / 2.0
    fields = tuple(Field(f"b{k}", ("t",)) for k in range(1, modes + 1))
    couplings = burgers_couplings(modes)
    equations = []
    for k in range(1, modes + 1):
        terms = [
            Term(1.0, (Factor(f"b{k}", (1,)),)),
            Term(half * viscosity * k**2, (Factor(f"b{k}", (0,)),)),
        ]
        for (target, j, m), value in couplings.items():
            if target == k:
                factors = (Factor(f"b{j}", (0,)), Factor(f"b{m}", (0,)))
                terms.append(Term(-half * value, factors))
        equations.append(Equation(tuple(terms)))
    data = tuple(
        Datum(f"b{k}", "t", -1.0, repr(float(amplitude)) if k == 1 else "0")
        for k in range(1, modes + 1)
    )
    return Problem(
        name=name,
        axes=(Axis("t", "time"),),
        fields=fields,
        equations=tuple(equations),
        constraints=(Constraint("ratio", "b1", "t", -1.0),),
        data=data,
        regular=Regular("b1", (-1.0,)),
        reference=Reference("ivp", tolerance=tolerance),
        description=(
            f"viscous Burgers u_s + u u_x = nu u_xx, u(x, 0) = a sin x, a = {amplitude:g}, "
            f"nu = {viscosity:g}, s in [0, {time:g}], {modes} sine modes"
        ),
    )


def cole_hopf(amplitude: float, viscosity: float) -> Callable[..., NDArray[np.float64]]:
    r"""Return the exact solution :math:`u(x, s)` of viscous Burgers from :math:`a\sin x`.

    The Cole–Hopf series of the module documentation, summed until its terms fall below
    rounding; vectorised over broadcasting ``x`` and ``s``.
    """
    beta = amplitude / (2.0 * viscosity)
    orders = [1]
    while special.ive(orders[-1], abs(beta)) > _SERIES_TOLERANCE * special.ive(0, abs(beta)):
        orders.append(orders[-1] + 1)
    n = np.asarray(orders, dtype=np.float64)
    # Scaled Bessel functions keep a large beta finite; the common factor e^|beta| cancels.
    weights = special.ive(n, beta)
    base = float(special.ive(0, beta))

    def solution(x: ArrayLike, s: ArrayLike) -> NDArray[np.float64]:
        x_, s_ = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(s, np.float64))
        decay = weights * np.exp(-viscosity * n**2 * s_[..., None])
        numerator = 4.0 * viscosity * np.sum(n * decay * np.sin(n * x_[..., None]), axis=-1)
        denominator = base + 2.0 * np.sum(decay * np.cos(n * x_[..., None]), axis=-1)
        return np.asarray(numerator / denominator)

    return solution


def sine_modes(
    solution: Callable[..., NDArray[np.float64]], modes: int, s: ArrayLike, points: int = 1024
) -> NDArray[np.float64]:
    r"""Return the sine coefficients :math:`b_k(s) = \frac1\pi\int_0^{2\pi} u\sin kx\,dx`.

    By the trapezoidal rule on ``points`` equally spaced points, exact for a trigonometric
    polynomial of degree below ``points``. The result has the shape of ``s`` then one axis of
    length ``modes``.
    """
    x = 2.0 * np.pi * np.arange(points) / points
    s_ = np.asarray(s, dtype=np.float64)
    values = solution(x, s_[..., None])
    k = np.arange(1, modes + 1)
    return np.asarray(2.0 * values @ np.sin(np.outer(x, k)) / points)


# ---------------------------------------------------------------------------- Navier–Stokes

#: A real basis function of the vorticity, :math:`\cos` or :math:`\sin` of :math:`k\cdot\mathbf{x}`,
#: as ``(kind, k_x, k_y)``.
Mode: TypeAlias = tuple[Literal["cos", "sin"], int, int]

#: The Galerkin products of the Jacobian's two factors (module documentation): for
#: :math:`h_a(A)\,h_b(B)`, each mode of :math:`A + B` (``+1``) or :math:`A - B` (``-1``) with the
#: sign it enters with, the common :math:`\tfrac12` apart.
_PRODUCTS: dict[tuple[str, str], tuple[tuple[Literal["cos", "sin"], int, int], ...]] = {
    ("sin", "sin"): (("cos", -1, 1), ("cos", 1, -1)),
    ("cos", "cos"): (("cos", -1, 1), ("cos", 1, 1)),
    ("sin", "cos"): (("sin", 1, 1), ("sin", -1, 1)),
    ("cos", "sin"): (("sin", 1, 1), ("sin", -1, -1)),
}

#: The vortex pair's :math:`\sigma^2`; its centres are :math:`(3\pi/4, \pi)` and
#: :math:`(5\pi/4, \pi)`.
_PAIR_WIDTH = Fraction(1, 4)

#: Relative margin within which two data are a tie for the anchor, which then goes to the first.
_TIE = 1.0e-9


def vorticity_modes(cutoff: int) -> tuple[Mode, ...]:
    r"""Return the real basis on the wavevectors :math:`0 < |k|^2 \le K_c` of the half-plane.

    Ordered by :math:`|k|^2`, then :math:`k_x`, then :math:`k_y`, each wavevector's cosine before
    its sine. :math:`K_c = 4` keeps 12 modes, 5 keeps 20 and 9 keeps 28.
    """
    if cutoff < 1:
        raise ValueError(f"the cutoff |k|^2 must be at least 1, got {cutoff}")
    radius = math.isqrt(cutoff)
    vectors = sorted(
        (
            (kx, ky)
            for kx in range(radius + 1)
            for ky in range(-radius, radius + 1)
            if 0 < kx * kx + ky * ky <= cutoff and (kx > 0 or ky > 0)
        ),
        key=lambda k: (k[0] ** 2 + k[1] ** 2, k[0], k[1]),
    )
    kinds: tuple[Literal["cos", "sin"], ...] = ("cos", "sin")
    return tuple((kind, kx, ky) for kx, ky in vectors for kind in kinds)


def mode_name(mode: Mode) -> str:
    """Return a mode's field name: ``("sin", 1, -2)`` is ``"s_1_m2"``."""
    kind, kx, ky = mode

    def part(value: int) -> str:
        return f"m{-value}" if value < 0 else str(value)

    return f"{kind[0]}_{part(kx)}_{part(ky)}"


def _half_plane(kx: int, ky: int) -> tuple[int, int, int]:
    """Return a wavevector folded into the half-plane, and the sign a sine takes there."""
    if kx > 0 or (kx == 0 and ky > 0):
        return kx, ky, 1
    return -kx, -ky, -1


def vorticity_couplings(cutoff: int) -> dict[tuple[int, int, int], Fraction]:
    r"""Return the Galerkin coefficients :math:`c_{kab}` of :math:`-J(\psi, \omega)`, exactly.

    :math:`(-J)_k = \sum_{a \le b} c_{kab}\,w_a w_b`, each unordered pair once with both orders
    of the Jacobian summed, the indices those of :func:`vorticity_modes` (module documentation).
    Every coefficient is rational.
    """
    modes = vorticity_modes(cutoff)
    index = {mode: i for i, mode in enumerate(modes)}
    couplings: dict[tuple[int, int, int], Fraction] = {}
    for a, (kind_a, px, py) in enumerate(modes):
        for b, (kind_b, qx, qy) in enumerate(modes):
            cross = px * qy - py * qx
            if not cross:
                continue
            sign = (-1 if kind_a == "cos" else 1) * (-1 if kind_b == "cos" else 1)
            scale = Fraction(sign * cross, 2 * (px * px + py * py))
            factors = ("sin" if kind_a == "cos" else "cos", "sin" if kind_b == "cos" else "cos")
            for kind, side, entry in _PRODUCTS[factors]:
                kx, ky, fold = _half_plane(px + side * qx, py + side * qy)
                if kx == ky == 0:
                    continue  # a constant from a cosine, which the sum cancels; a sine is zero
                target = index.get((kind, kx, ky))
                if target is None:
                    continue  # outside the truncation
                key = (target, min(a, b), max(a, b))
                weight = entry * fold if kind == "sin" else entry
                couplings[key] = couplings.get(key, Fraction(0)) - weight * scale
    return {key: value for key, value in couplings.items() if value}


def navier_stokes(
    cutoff: int,
    initial: Mapping[Mode, str],
    viscosity: float,
    time: float,
    *,
    name: str = "R10",
    reference: Reference | None = None,
    description: str = "",
    tolerance: float = 1.0e-13,
) -> Problem:
    r"""Return the Galerkin truncation of two-dimensional Navier–Stokes (module documentation).

    Parameters
    ----------
    cutoff
        :math:`K_c`, the largest :math:`|k|^2` kept.
    initial
        The initial vorticity's coefficient of each mode, as the text of a constant expression
        (the recipe that defines it); a mode left out starts at zero.
    viscosity, time
        :math:`\nu`, and :math:`T`, the end of the time interval.
    name, description
        The problem's name and free text.
    reference
        The reference; by default the system's initial-value problem, to ``tolerance``.
    tolerance
        The initial-value reference's tolerance.

    Returns
    -------
    Problem
        One field per mode (:func:`mode_name`) on the time axis ``t``, one equation each, and a
        ratio constraint at :math:`t = -1` anchored on the mode with the largest datum (the
        first of a tie), which is the regular point.
    """
    modes = vorticity_modes(cutoff)
    if set(initial) - set(modes):
        raise ValueError(f"modes outside the truncation: {sorted(set(initial) - set(modes))}")
    if viscosity <= 0 or time <= 0:
        raise ValueError("Navier-Stokes needs positive nu and T")
    names = [mode_name(mode) for mode in modes]
    half = time / 2.0
    driven: dict[int, list[Term]] = {c: [] for c in range(len(modes))}
    for (c, a, b), value in vorticity_couplings(cutoff).items():
        factors = (Factor(names[a], (0,)), Factor(names[b], (0,)))
        driven[c].append(Term(-half * float(value), factors))
    equations = []
    for c, (_, kx, ky) in enumerate(modes):
        linear = [
            Term(1.0, (Factor(names[c], (1,)),)),
            Term(half * viscosity * (kx * kx + ky * ky), (Factor(names[c], (0,)),)),
        ]
        equations.append(Equation((*linear, *driven[c])))
    texts = [initial.get(mode, "0") for mode in modes]
    sizes = np.abs([float(compile_expression(text, [])()) for text in texts])
    if not sizes.any():
        raise ValueError("the initial vorticity is zero")
    anchor = names[int(np.flatnonzero(sizes >= (1.0 - _TIE) * sizes.max())[0])]
    return Problem(
        name=name,
        axes=(Axis("t", "time"),),
        fields=tuple(Field(field, ("t",)) for field in names),
        equations=tuple(equations),
        constraints=(Constraint("ratio", anchor, "t", -1.0),),
        data=tuple(Datum(field, "t", -1.0, text) for field, text in zip(names, texts, strict=True)),
        regular=Regular(anchor, (-1.0,)),
        reference=reference or Reference("ivp", tolerance=tolerance),
        description=description,
    )


def taylor_green(cutoff: int, viscosity: float, time: float, *, name: str = "R10a") -> Problem:
    r"""Return Navier–Stokes from the Taylor–Green vortex :math:`\omega_0 = 2\sin x\sin y`.

    :math:`2\sin x\sin y = \cos(x - y) - \cos(x + y)`, two modes of :math:`|k|^2 = 2`, whose
    Jacobian vanishes, so :math:`\omega = \omega_0 e^{-2\nu s}` exactly: the reference is that
    closed form, :math:`e^{-\nu T(t + 1)}` on the Chebyshev axis, in every field.
    """
    if cutoff < 2:
        raise ValueError("the Taylor-Green vortex needs |k|^2 = 2 in the truncation")
    signs: dict[Mode, str] = {("cos", 1, -1): "", ("cos", 1, 1): "-"}
    decay = f"exp(-{float(viscosity * time)!r}*(t + 1))"
    expressions = tuple(
        f"{signs[mode]}{decay}" if mode in signs else "0" for mode in vorticity_modes(cutoff)
    )
    return navier_stokes(
        cutoff,
        {mode: f"{sign}1" for mode, sign in signs.items()},
        viscosity,
        time,
        name=name,
        reference=Reference("expression", expressions=expressions),
        description=(
            f"2-D Navier-Stokes from the Taylor-Green vortex 2 sin x sin y, nu = {viscosity:g}, "
            f"s in [0, {time:g}], |k|^2 <= {cutoff}"
        ),
    )


def vortex_pair(
    cutoff: int, strength: float, viscosity: float, time: float, *, name: str = "R10"
) -> Problem:
    r"""Return Navier–Stokes from a counter-rotating pair of Gaussian vortices of norm ``strength``.

    Vortices :math:`\pm e^{-|\mathbf{x} - c|^2/2\sigma^2}` with :math:`\sigma^2 = 1/4` at
    :math:`c = (3\pi/4, \pi)` and :math:`(5\pi/4, \pi)`, periodised and projected exactly on the
    modes: a periodised Gaussian's coefficient of :math:`e^{ik\cdot\mathbf{x}}` is
    :math:`(\sigma^2/2\pi)\,e^{-\sigma^2|k|^2/2}e^{-ik\cdot c}`, so the pair's cosines cancel and
    its sines are proportional to :math:`(-1)^{k_x + k_y + 1}\sin(\pi k_x/4)\,e^{-|k|^2/8}`. They
    are scaled so that the coefficients' norm is ``strength``, :math:`\|w_0\| = a`, as R9's
    :math:`a` is; each datum is that closed form as text, so the problem is its recipe.
    """
    if strength <= 0:
        raise ValueError(f"the pair's strength must be positive, got {strength}")
    shapes: dict[Mode, str] = {}
    squares: list[str] = []
    for mode in vorticity_modes(cutoff):
        kind, kx, ky = mode
        if kind == "cos" or kx % 4 == 0:
            continue
        k2 = kx * kx + ky * ky
        sign = "-" if (kx + ky) % 2 == 0 else ""
        shapes[mode] = f"{sign}sin({kx}*pi/4)*exp(-{k2 * _PAIR_WIDTH / 2})"
        squares.append(f"sin({kx}*pi/4)**2*exp(-{k2 * _PAIR_WIDTH})")
    norm = f"sqrt({' + '.join(squares)})"
    initial = {mode: f"{float(strength)!r}*{shape}/{norm}" for mode, shape in shapes.items()}
    return navier_stokes(
        cutoff,
        initial,
        viscosity,
        time,
        name=name,
        description=(
            f"2-D Navier-Stokes from a counter-rotating vortex pair of norm a = {strength:g}, "
            f"nu = {viscosity:g}, s in [0, {time:g}], |k|^2 <= {cutoff}"
        ),
    )


def vorticity_field(
    modes: Sequence[Mode], coefficients: ArrayLike, x: ArrayLike, y: ArrayLike
) -> NDArray[np.float64]:
    r"""Return :math:`\omega(x, y) = \sum w_m \varphi_m`, broadcasting ``x`` and ``y``."""
    x_, y_ = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    values = np.zeros(x_.shape)
    for (kind, kx, ky), w in zip(modes, np.asarray(coefficients, np.float64), strict=True):
        phase = kx * x_ + ky * y_
        values += w * (np.cos(phase) if kind == "cos" else np.sin(phase))
    return values


def vorticity_dns(
    cutoff: int,
    initial: ArrayLike,
    viscosity: float,
    times: ArrayLike,
    *,
    points: int = 48,
    tolerance: float = 1.0e-11,
) -> NDArray[np.float64]:
    r"""Return the equation's own solution on the truncation's modes, by pseudo-spectral simulation.

    The vorticity equation itself, not its truncation, from the truncation's data ``initial``
    (the coefficients of :func:`vorticity_modes`): Fourier coefficients on a ``points``-square
    grid, the Jacobian formed on the grid with the two-thirds rule's dealiasing, integrated by
    ``solve_ivp`` (DOP853) to ``tolerance``. Against the Galerkin system's reference it measures
    the truncation's error, apart from the lift's and the time axis's.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times), len(modes))``: the coefficients at each time :math:`s` of ``times``
        (increasing, from 0).
    """
    modes = vorticity_modes(cutoff)
    w0 = np.asarray(initial, dtype=np.float64)
    s = np.asarray(times, dtype=np.float64)
    wavenumbers = np.fft.fftfreq(points, 1.0 / points)
    kx, ky = np.meshgrid(wavenumbers, wavenumbers, indexing="ij")
    k2 = kx**2 + ky**2
    inverse = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 > 0)
    kept = (np.abs(kx) < points / 3) & (np.abs(ky) < points / 3)
    area = float(points * points)
    spectrum = np.zeros((points, points), dtype=np.complex128)
    for (kind, mx, my), w in zip(modes, w0, strict=True):
        half = area * w / 2 if kind == "cos" else -1j * area * w / 2
        spectrum[mx % points, my % points] += half
        spectrum[-mx % points, -my % points] += np.conj(half)
    size = points * points

    def slopes(_: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        omega = (state[:size] + 1j * state[size:]).reshape(points, points)
        stream = omega * inverse
        psi_x, psi_y = (np.fft.ifft2(1j * k * stream).real for k in (kx, ky))
        omega_x, omega_y = (np.fft.ifft2(1j * k * omega).real for k in (kx, ky))
        jacobian = np.fft.fft2(psi_x * omega_y - psi_y * omega_x)
        slope = (-jacobian - viscosity * k2 * omega) * kept
        return np.concatenate([slope.real.ravel(), slope.imag.ravel()])

    start = np.concatenate([spectrum.real.ravel(), spectrum.imag.ravel()])
    solution = integrate.solve_ivp(
        slopes,
        (0.0, float(s[-1])),
        start,
        method="DOP853",
        rtol=tolerance,
        atol=tolerance * float(np.max(np.abs(start))),
        t_eval=s,
    )
    if not solution.success:
        raise RuntimeError(f"the vorticity simulation failed: {solution.message}")
    result = np.empty((s.size, len(modes)))
    for j in range(s.size):
        omega = solution.y[:size, j] + 1j * solution.y[size:, j]
        for m, (kind, mx, my) in enumerate(modes):
            value = omega[(mx % points) * points + my % points] * 2.0 / area
            result[j, m] = value.real if kind == "cos" else -value.imag
    return result


# ---------------------------------------------------------------------------- the structured form


@dataclass(frozen=True)
class Symbol:
    r"""A Fourier multiplier, :math:`c\,\prod_d (ik_d)^{a_d}\,|k|^{2e}`.

    Attributes
    ----------
    coefficient
        :math:`c`.
    derivatives
        :math:`a_d`, the order of the derivative along each axis (none for a constant).
    laplacian
        :math:`e`, the power of :math:`|k|^2`; negative for an inverse, which is zero at
        :math:`k = 0`.
    """

    coefficient: float = 1.0
    derivatives: tuple[int, ...] = ()
    laplacian: int = 0

    @property
    def phase(self) -> complex:
        r"""Return :math:`i^{\sum_d a_d}`, the symbol's phase, the same at every :math:`k`."""
        return complex(1j ** (sum(self.derivatives) % 4))

    def magnitude(self, k: Sequence[int]) -> float:
        r"""Return the real :math:`c\,\prod_d k_d^{a_d}|k|^{2e}`, the symbol over its phase."""
        size = sum(value * value for value in k)
        if self.laplacian < 0 and size == 0:
            return 0.0
        value = self.coefficient * float(size) ** self.laplacian
        for order, component in zip(self.derivatives, k, strict=False):
            value *= component**order
        return float(value)

    def value(self, k: Sequence[int]) -> complex:
        """Return the symbol at the wavevector ``k``."""
        return self.phase * self.magnitude(k)


@dataclass(frozen=True)
class Product:
    r"""One quadratic term, :math:`\text{output}((\text{left}\,u)\,(\text{right}\,u))`."""

    left: Symbol
    right: Symbol
    output: Symbol


@dataclass(frozen=True)
class Galerkin:
    r"""An equation :math:`u_s = L_1 u + \sum_\text{products}` on a Fourier truncation.

    Attributes
    ----------
    wavevectors
        The truncation: every wavevector kept, both signs.
    linear
        :math:`L_1`'s symbol.
    products
        The quadratic terms, each a pointwise product between multipliers.
    time
        :math:`T`: on the Chebyshev axis every term carries :math:`T/2`.
    basis
        The problem's real basis: ``"sine"`` (:math:`\sin kx`, R9) or ``"real"`` (the cosines
        and sines of :func:`vorticity_modes`, R10).
    """

    wavevectors: tuple[tuple[int, ...], ...]
    linear: Symbol
    products: tuple[Product, ...]
    time: float
    basis: Literal["sine", "real"]

    @property
    def axes(self) -> int:
        """The number of periodic axes."""
        return len(self.wavevectors[0])

    @property
    def largest(self) -> int:
        """The largest :math:`|k_d|` of the truncation, on any axis."""
        return max(abs(value) for k in self.wavevectors for value in k)


def burgers_galerkin(modes: int, viscosity: float, time: float) -> Galerkin:
    r"""Return :func:`burgers`' structured form, :math:`u_s = \nu u_{xx} - \partial_x(u^2)/2`."""
    return Galerkin(
        wavevectors=tuple((sign * k,) for k in range(1, modes + 1) for sign in (1, -1)),
        linear=Symbol(-viscosity, laplacian=1),
        products=(Product(Symbol(), Symbol(), Symbol(-0.5, (1,))),),
        time=time,
        basis="sine",
    )


def vorticity_galerkin(cutoff: int, viscosity: float, time: float) -> Galerkin:
    r"""Return :func:`navier_stokes`' structured form (module documentation).

    :math:`\omega_s = \nu\nabla^2\omega - (\psi_x\omega_y - \psi_y\omega_x)`: with
    :math:`\hat\psi = \hat\omega/|k|^2`, :math:`\psi_x` is the multiplier :math:`ik_x/|k|^2` of
    :math:`\omega`.
    """
    halves = [(kx, ky) for _, kx, ky in vorticity_modes(cutoff)[::2]]
    return Galerkin(
        wavevectors=tuple(k for kx, ky in halves for k in ((kx, ky), (-kx, -ky))),
        linear=Symbol(-viscosity, laplacian=1),
        products=(
            Product(Symbol(1.0, (1, 0), -1), Symbol(1.0, (0, 1)), Symbol(-1.0)),
            Product(Symbol(1.0, (0, 1), -1), Symbol(1.0, (1, 0)), Symbol(1.0)),
        ),
        time=time,
        basis="real",
    )


def fourier_field(galerkin: Galerkin) -> list[sp.csr_array]:
    r"""Return :math:`[F_1, F_2]` in the variables :math:`v = \sqrt2\,\hat u` on the truncation.

    :math:`F_1` is the linear symbol's diagonal; :math:`F_2`'s column :math:`(p, q)` (``p``'s
    index times the truncation's size, plus ``q``'s) holds, at row :math:`p + q` where it is kept,
    each product's :math:`\text{output}(p + q)\,\text{left}(p)\,\text{right}(q)`: the exact
    convolution, truncated. :math:`\hat u`'s product is :math:`v`'s over two, hence
    :math:`F_2`'s :math:`1/\sqrt2`. Both carry the Chebyshev axis's :math:`T/2`.
    """
    vectors = galerkin.wavevectors
    index = {k: i for i, k in enumerate(vectors)}
    size = len(vectors)
    half = galerkin.time / 2.0
    linear = sp.diags_array([half * galerkin.linear.value(k) for k in vectors]).astype(complex)
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for p, left in enumerate(vectors):
        for q, right in enumerate(vectors):
            target = index.get(tuple(a + b for a, b in zip(left, right, strict=True)))
            if target is None:
                continue
            entry = sum(
                product.output.value(vectors[target])
                * product.left.value(left)
                * product.right.value(right)
                for product in galerkin.products
            )
            if entry:
                rows.append(target)
                columns.append(p * size + q)
                values.append(half * entry / math.sqrt(2.0))
    quadratic = sp.csr_array((values, (rows, columns)), shape=(size, size * size), dtype=complex)
    return [sp.csr_array(linear), quadratic]


def real_embedding(galerkin: Galerkin) -> sp.csr_array:
    r"""Return :math:`E`, with :math:`v = Ew` for the problem's real coefficients :math:`w`.

    :math:`\cos(k\cdot\mathbf{x})` puts :math:`1/\sqrt2` on :math:`\pm k`, and
    :math:`\sin(k\cdot\mathbf{x})` puts :math:`\mp i/\sqrt2`, from
    :math:`\hat u_{\pm k} = (a \mp ib)/2`. :math:`E` is an isometry (:math:`E^\dagger E = I`),
    unitary for the real basis, which holds every wavevector.
    """
    index = {k: i for i, k in enumerate(galerkin.wavevectors)}
    columns: list[tuple[str, tuple[int, ...]]]
    if galerkin.basis == "sine":
        columns = [("sin", (k,)) for k in range(1, len(index) // 2 + 1)]
    else:
        columns = [(kind, (kx, ky)) for kind, kx, ky in vorticity_modes(_cutoff(galerkin))]
    E = np.zeros((len(index), len(columns)), dtype=complex)
    root = 1.0 / math.sqrt(2.0)
    for column, (kind, k) in enumerate(columns):
        minus = tuple(-value for value in k)
        if kind == "cos":
            E[index[k], column] = E[index[minus], column] = root
        else:
            E[index[k], column], E[index[minus], column] = -1j * root, 1j * root
    return sp.csr_array(E)


def _cutoff(galerkin: Galerkin) -> int:
    """Return the largest :math:`|k|^2` of a truncation."""
    return max(sum(value * value for value in k) for k in galerkin.wavevectors)
