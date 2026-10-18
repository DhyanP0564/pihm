r"""QITE against the minimax edge filter at like error.

Two filters prepare the ground state from the same walk: imaginary time,
:math:`f_\tau(x)=e^{-\tau(1+x)}` truncated at degree :math:`d` (:mod:`pihm.qsp.filters`), and the
minimax edge filter :math:`p_d` (:mod:`pihm.qsp.minimax`). Both are anchored at :math:`x=-1`, so
:math:`\sum_k|p_k|=1` for each and GQSP realises them without rescaling. A degree is only
comparable at a common error, and this module fixes it three ways, from the weakest assumption to
the strongest:

* **Equal suppression** :math:`\eta`, the excited interval :math:`[x_1,1]` relative to the ground
  state :math:`x_0` (:func:`compare_suppression`). It bounds the infidelity of every initial state
  with ground overlap :math:`\gamma` by :math:`\eta^2(1-\gamma^2)/\gamma^2`, and needs only
  :math:`x_0` and :math:`x_1`.
* **Equal certified infidelity** :math:`\varepsilon`, the same bound solved for
  :math:`\eta=\gamma\sqrt{\varepsilon/(1-\gamma^2)}` at a problem's own :math:`\gamma`.
* **Equal measured infidelity** :math:`\varepsilon` on a problem's actual spectrum and initial
  state (:func:`compare_state`).

For QITE every level is one condition. The truncation leaves a remainder of norm at most the tail
:math:`T=\sum_{k>d}|p_k|`, so if the untruncated filter leaves a component :math:`e` off the
target and :math:`g` along it, the truncated one's ratio is at most :math:`(e+T)/(g-T)`. That is
at most :math:`s` whenever

.. math:: T \le \frac{s\,g-e}{1+s},

which fixes the certified degree at each :math:`\tau` (:func:`pihm.qsp.filters.qite_coefficients`),
provided the tail asked for is at least :data:`CERTIFIABLE_TAIL`. Above a nonzero ground eigenvalue
:math:`g` falls as :math:`e^{-\tau(1+x_0)}`, and where the tail it needs is smaller than that, no
double-precision filter is certified and QITE is reported as not reaching the error.
QITE's degree is then minimised over :math:`\tau`, which has no closed form. The search is a
logarithmic grid above the least feasible :math:`\tau`, refined linearly about its best point, so
QITE's reported degree is an upper bound on its optimum, tight to the refinement's spacing. The
minimax filter needs no search: it is optimal at the first level by Chebyshev's theorem, so at
equal suppression QITE can never use fewer degrees. The comparison measures by how much.

At the third level, a degree counts only if every larger degree up to the certified one also
reaches :math:`\varepsilon`. The minimax filter oscillates on :math:`[x_1,1]` and a truncated
series oscillates about its limit, so either can dip below :math:`\varepsilon` early by
coincidence. That coincidence would not survive a small change of spectrum.

Measured QITE degrees are sought only where the filter's weight on the target,
:math:`\gamma^2e^{-2\tau(1+x_0)}`, is at least
:data:`~pihm.qsp.prepare.RESOLVABLE_PROBABILITY`. Above a nonzero ground eigenvalue that weight
falls as :math:`\tau` grows, and below that level the partial sums' rounding exceeds the tail the
certificate needs, so no measurement resolves it; the pipeline counts such a run as not
preparable. The minimax filter's success probability is reported against the same level.

Values in the ground cluster are formed from differences of singular values,
:math:`x_j-x_0=2(s_j^2-s_0^2)/\alpha_R^2`, and Chebyshev terms from angles measured from
:math:`-1`, :math:`T_k(x_j)=(-1)^k\cos(k\phi_j)` with :math:`\phi_j=2\arcsin(s_j/\alpha_R)`,
which keeps full relative precision where the spectrum crowds against :math:`-1`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pihm.qsp.filters import MAX_DEGREE, qite_coefficients
from pihm.qsp.minimax import minimax_degree, minimax_values
from pihm.qsp.prepare import RESOLVABLE_PROBABILITY

__all__ = [
    "CERTIFIABLE_TAIL",
    "INFIDELITIES",
    "SUPPRESSIONS",
    "FilterChoice",
    "Spectrum",
    "StateComparison",
    "SuppressionComparison",
    "compare_state",
    "compare_suppression",
    "preparations",
    "qite_state_coefficients",
    "qite_suppression_degree",
    "spectrum",
]

#: The suppressions the synthetic sweep compares at.
SUPPRESSIONS: Final[tuple[float, ...]] = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)

#: The least truncation tail a certificate may claim. A filter with :math:`\sum_k|p_k|=1` is
#: formed and evaluated in double precision to about this absolute accuracy (the coefficients'
#: own, ``pihm.qsp.filters._exp_scaled_bessel_i``, is about :math:`5\times10^{-17}`), so a
#: smaller tail would certify rounding.
CERTIFIABLE_TAIL: Final[float] = 1e-15

#: The infidelities the problem study compares at.
INFIDELITIES: Final[tuple[float, ...]] = (1e-4, 1e-6, 1e-8)

#: The logarithmic grid of :math:`\tau/\tau_{\min}-1` QITE's search starts from, and the number
#: of points its linear refinement adds between the best point's neighbours.
_TAU_OFFSETS: Final[NDArray[np.float64]] = np.logspace(-4.0, 2.0, 25)
_REFINE: Final[int] = 16

#: Degrees per block when partial sums are streamed.
_CHUNK: Final[int] = 2048


def preparations(probability: float) -> int:
    r"""Return the preparations amplitude amplification applies at success probability ``p``.

    That is :math:`2k+1`, :math:`k=\lfloor\pi/(4\arcsin\sqrt p)\rfloor`, as
    :func:`~pihm.qsp.prepare.t3` counts them.
    """
    if probability <= 0.0:
        raise ValueError(f"preparations needs a positive probability, got {probability}")
    theta = math.asin(min(math.sqrt(probability), 1.0))
    return 2 * math.floor(math.pi / (4.0 * theta)) + 1


@dataclass(frozen=True)
class FilterChoice:
    r"""One filter at one degree.

    Attributes
    ----------
    degree
        The degree :math:`d`.
    tau
        QITE's imaginary time; ``nan`` for the minimax filter.
    ground
        The filter's value on the ground state, :math:`|P(x_0)|`, so that the success
        probability is about :math:`\gamma^2|P(x_0)|^2`.
    """

    degree: int
    tau: float
    ground: float


def _certified_degree(tau: float, e: float, g: float, s: float, max_degree: int) -> int | None:
    """Return the least degree whose tail certifies ``(e + T) / (g - T) <= s``, or ``None``."""
    tolerance = (s * g - e) / (1.0 + s)
    if not CERTIFIABLE_TAIL <= tolerance < 1.0:
        return None
    try:
        return qite_coefficients(tau, tolerance=tolerance, max_degree=max_degree).degree
    except ValueError:  # past max_degree, or below the coefficients' double-precision floor
        return None


def _search(tau_floor: float, degree: Callable[[float], int | None]) -> tuple[float, int] | None:
    """Minimise ``degree(tau)`` over ``tau > tau_floor``: a logarithmic grid, refined linearly."""
    taus = tau_floor * (1.0 + _TAU_OFFSETS)
    found = [degree(tau) for tau in taus]
    feasible = [i for i, d in enumerate(found) if d is not None]
    if not feasible:
        return None
    best = min(feasible, key=lambda i: (found[i], i))
    low = taus[best - 1] if best > 0 else tau_floor * (1.0 + 0.5 * _TAU_OFFSETS[0])
    high = taus[min(best + 1, taus.size - 1)]
    best_degree = found[best]
    assert best_degree is not None
    candidates = [(float(taus[best]), best_degree)]
    for tau in np.linspace(low, high, _REFINE + 2)[1:-1]:
        d = degree(float(tau))
        if d is not None:
            candidates.append((float(tau), d))
    return min(candidates, key=lambda c: (c[1], c[0]))


def qite_suppression_degree(
    x_ground: float, x_excited: float, suppression: float, *, max_degree: int = MAX_DEGREE
) -> FilterChoice | None:
    r"""Return QITE's least certified degree reaching ``suppression``, over :math:`\tau`.

    The untruncated filter's suppression is :math:`e^{-\tau(x_1-x_0)}`, so :math:`\tau` must pass
    :math:`\ln(1/\eta)/(x_1-x_0)`; above it the tail must satisfy the module docstring's
    condition with :math:`e=e^{-\tau(1+x_1)}`, :math:`g=e^{-\tau(1+x_0)}` and :math:`s=\eta`.
    Returns ``None`` if no :math:`\tau` on the search grid reaches it within ``max_degree`` in
    double precision.
    """
    gap = x_excited - x_ground
    if not gap > 0.0:
        raise ValueError(f"the excited interval must lie above the ground state, gap {gap}")

    def degree(tau: float) -> int | None:
        g = math.exp(-tau * (1.0 + x_ground))
        e = math.exp(-tau * (1.0 + x_excited))
        return _certified_degree(tau, e, g, suppression, max_degree)

    found = _search(math.log(1.0 / suppression) / gap, degree)
    if found is None:
        return None
    tau, d = found
    return FilterChoice(degree=d, tau=tau, ground=math.exp(-tau * (1.0 + x_ground)))


@dataclass(frozen=True)
class SuppressionComparison:
    r"""QITE and the minimax filter at one suppression (the first level of the module docstring).

    Attributes
    ----------
    x_ground, x_excited, suppression
        :math:`x_0`, :math:`x_1` and :math:`\eta`.
    minimax
        The minimax filter at its least degree.
    qite
        QITE at its least certified degree over :math:`\tau`, or ``None`` if none is reachable.
    production
        QITE as the pipeline builds it for :math:`\varepsilon_f=\eta`:
        :math:`\tau=\ln(1/\eta)/(x_1-x_0)` and tail :math:`\eta/10`
        (:func:`~pihm.qsp.filters.qite_filter`), or ``None`` past ``max_degree``.
    production_suppression
        That filter's certified suppression, :math:`(e+T)/(g-T)`; ``inf`` where the tail
        reaches the ground value, so nothing is certified.
    """

    x_ground: float
    x_excited: float
    suppression: float
    minimax: FilterChoice
    qite: FilterChoice | None
    production: FilterChoice | None
    production_suppression: float


def compare_suppression(
    x_ground: float, x_excited: float, suppression: float, *, max_degree: int = MAX_DEGREE
) -> SuppressionComparison:
    """Compare QITE and the minimax filter at equal suppression (module docstring)."""
    d = minimax_degree(x_ground, x_excited, suppression)
    ground = float(abs(minimax_values(np.array([x_ground]), d, x_excited)[0]))
    minimax = FilterChoice(degree=d, tau=math.nan, ground=ground)
    qite = qite_suppression_degree(x_ground, x_excited, suppression, max_degree=max_degree)

    tau = math.log(1.0 / suppression) / (x_excited - x_ground)
    production: FilterChoice | None = None
    production_suppression = math.inf
    try:
        built = qite_coefficients(tau, tolerance=suppression / 10.0, max_degree=max_degree)
    except ValueError:
        pass
    else:
        g = math.exp(-tau * (1.0 + x_ground))
        e = math.exp(-tau * (1.0 + x_excited))
        production = FilterChoice(degree=built.degree, tau=tau, ground=g)
        if g > built.tail_bound:
            production_suppression = (e + built.tail_bound) / (g - built.tail_bound)
    return SuppressionComparison(
        x_ground=x_ground,
        x_excited=x_excited,
        suppression=suppression,
        minimax=minimax,
        qite=qite,
        production=production,
        production_suppression=production_suppression,
    )


@dataclass(frozen=True)
class Spectrum:
    r"""The spectrum of :math:`H'=2H/\alpha_R^2-1` as an initial state sees it.

    Attributes
    ----------
    singular
        :math:`s_j`, the singular values of :math:`R` (zeros appended to its column count),
        ascending, so :math:`x_j=2s_j^2/\alpha_R^2-1` ascends.
    alpha_r
        The block encoding's subnormalisation.
    weights
        :math:`|\langle v_j|\varphi_0\rangle|^2`, the initial state's weight on each eigenvector.
    target
        How many of the lowest eigenvectors the filter keeps: 1, or the kernel's dimension in a
        doubled space, where the target is :math:`\varphi_0`'s projection.
    """

    singular: NDArray[np.float64]
    alpha_r: float
    weights: NDArray[np.float64]
    target: int

    @property
    def values(self) -> NDArray[np.float64]:
        """:math:`x_j`."""
        result: NDArray[np.float64] = 2.0 * (self.singular / self.alpha_r) ** 2 - 1.0
        return result

    @property
    def angles(self) -> NDArray[np.float64]:
        r""":math:`\phi_j=\pi-\arccos x_j=2\arcsin(s_j/\alpha_R)`, angles measured from -1."""
        result: NDArray[np.float64] = 2.0 * np.arcsin(np.clip(self.singular / self.alpha_r, 0, 1))
        return result

    @property
    def gamma(self) -> float:
        r""":math:`\gamma`, the initial state's overlap with the target."""
        return float(math.sqrt(float(np.sum(self.weights[: self.target]))))

    @property
    def x_ground(self) -> float:
        """The largest target eigenvalue, :math:`x_0`."""
        return float(self.values[self.target - 1])

    @property
    def x_excited(self) -> float:
        """The least eigenvalue above the target, :math:`x_1`."""
        return float(self.values[self.target])

    @property
    def offset(self) -> float:
        r""":math:`(1+x_0)/(x_1-x_0)=\lambda_0/\Delta`, the ground's height above -1 in gaps."""
        s0, s1 = self.singular[self.target - 1], self.singular[self.target]
        return float(s0**2 / (s1**2 - s0**2))

    def infidelity(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Return the prepared state's infidelity for filter values at the eigenvalues.

        ``values`` is :math:`P(x_j)` along its last axis, any leading shape. With target weight
        :math:`\gamma^2=\sum_{j<t}w_j` and :math:`\bar P=\sum_{j<t}w_jP_j/\gamma^2`, the target
        is the initial state's projection, and

        .. math:: 1-F=\frac{\sum_{j<t}w_j(P_j-\bar P)^2+\sum_{j\ge t}w_jP_j^2}{\sum_jw_jP_j^2},

        formed without subtracting from 1.
        """
        t = self.target
        w = self.weights
        head = values[..., :t]
        mean = (head @ w[:t]) / float(np.sum(w[:t]))
        spread = ((head - mean[..., None]) ** 2) @ w[:t]
        excited = (values[..., t:] ** 2) @ w[t:]
        total = (values**2) @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            result: NDArray[np.float64] = (spread + excited) / total
        return result

    def probability(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Return :math:`\sum_jw_jP(x_j)^2`, the success probability."""
        result: NDArray[np.float64] = (values**2) @ self.weights
        return result


def spectrum(
    matrix: ArrayLike,
    alpha_r: float,
    phi0: ArrayLike,
    target: int,
) -> Spectrum:
    """Return the :class:`Spectrum` of a dense residual matrix for an initial state."""
    dense = np.asarray(matrix)
    _u, s, vh = np.linalg.svd(dense, full_matrices=True)
    columns = dense.shape[1]
    singular = np.zeros(columns)
    singular[: s.size] = s
    order = np.argsort(singular, kind="stable")
    weights = np.abs(vh[order] @ np.asarray(phi0)) ** 2
    return Spectrum(
        singular=singular[order].astype(np.float64),
        alpha_r=float(alpha_r),
        weights=(weights / np.sum(weights)).astype(np.float64),
        target=int(target),
    )


def _minimax_ratios(spec: Spectrum, degrees: NDArray[np.int64]) -> NDArray[np.float64]:
    r"""Return :math:`p_d(x_j)/p_d(x_0)` for each degree (rows) and eigenvalue (columns).

    With :math:`z_j=2(s_1^2-s_j^2)/(\alpha_R^2-s_1^2)=-w(x_j)-1`: below :math:`x_1` the ratio is
    :math:`\cosh(d\,a_j)/\cosh(d\,a_0)`, :math:`a_j=\operatorname{arccosh}(1+z_j)`; on
    :math:`[x_1,1]` it is :math:`(-1)^d\cos(d\,b_j)/\cosh(d\,a_0)`,
    :math:`b_j=\pi-2\arcsin\sqrt{-z_j/2}=\arccos w(x_j)`.
    """
    s1 = spec.singular[spec.target]
    z = 2.0 * (s1**2 - spec.singular**2) / (spec.alpha_r**2 - s1**2)
    below = z > 0.0
    a = np.log1p(z[below] + np.sqrt(z[below] * (z[below] + 2.0)))
    b = np.pi - 2.0 * np.arcsin(np.sqrt(np.clip(-z[~below] / 2.0, 0.0, 1.0)))
    zg = z[spec.target - 1]
    a0 = float(np.log1p(zg + np.sqrt(zg * (zg + 2.0))))
    d = degrees.astype(np.float64)[:, None]
    out = np.empty((degrees.size, z.size))
    # cosh(u)/cosh(v) = exp(u - v)(1 + exp(-2u))/(1 + exp(-2v)), never forming either cosh
    denominator = 1.0 + np.exp(-2.0 * d * a0)
    out[:, below] = np.exp(d * (a - a0)) * (1.0 + np.exp(-2.0 * d * a)) / denominator
    sign = np.where(degrees % 2 == 1, -1.0, 1.0)[:, None]
    out[:, ~below] = sign * np.cos(d * b) * (2.0 * np.exp(-d * a0) / denominator)
    return out


def _minimax_ground(spec: Spectrum, degree: int) -> float:
    r"""Return :math:`p_d(x_0)=\cosh(d\,a_0)/\cosh(d\,a_{-1})`, the value at the ground state."""
    s0, s1 = spec.singular[spec.target - 1], spec.singular[spec.target]
    z0 = 2.0 * (s1**2 - s0**2) / (spec.alpha_r**2 - s1**2)
    z_edge = 2.0 * s1**2 / (spec.alpha_r**2 - s1**2)
    a0 = math.log1p(z0 + math.sqrt(z0 * (z0 + 2.0)))
    edge = math.log1p(z_edge + math.sqrt(z_edge * (z_edge + 2.0)))
    return (
        math.exp(degree * (a0 - edge))
        * (1.0 + math.exp(-2.0 * degree * a0))
        / (1.0 + math.exp(-2.0 * degree * edge))
    )


def _last_violation(
    infidelities: Callable[[int, int], NDArray[np.float64]], top: int, eps: float
) -> int:
    """Return the largest degree in ``[0, top]`` whose infidelity exceeds ``eps``, or -1."""
    last = -1
    for start in range(0, top + 1, _CHUNK):
        stop = min(start + _CHUNK, top + 1)
        above = np.nonzero(~(infidelities(start, stop) <= eps))[0]
        if above.size:
            last = start + int(above[-1])
    return last


def _minimax_state(spec: Spectrum, eps: float) -> tuple[int, int, float]:
    """Return the minimax filter's certified degree, measured degree and its infidelity."""
    gamma2 = spec.gamma**2
    if gamma2 >= 1.0:
        return 0, 0, 0.0
    eta = math.sqrt(eps * gamma2 / (1.0 - gamma2))
    certified = minimax_degree(spec.x_ground, spec.x_excited, min(eta, 0.5))

    def infidelities(start: int, stop: int) -> NDArray[np.float64]:
        return spec.infidelity(_minimax_ratios(spec, np.arange(start, stop)))

    measured = _last_violation(infidelities, certified, eps) + 1
    if measured > certified:
        raise RuntimeError(f"the minimax filter's certified degree {certified} misses {eps}")
    reached = float(infidelities(measured, measured + 1)[0])
    return certified, measured, reached


def _qite_values(
    spec: Spectrum,
    coefficients: NDArray[np.float64],
    start: int,
    stop: int,
    running: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Stream the partial sums of degrees ``start`` to ``stop - 1``, updating ``running``."""
    k = np.arange(start, stop)
    signed = coefficients[start:stop] * np.where(k % 2 == 1, -1.0, 1.0)  # T_k = (-1)^k cos(k phi)
    terms = signed[:, None] * np.cos(k[:, None] * spec.angles[None, :])
    partial = np.cumsum(terms, axis=0) + running
    running[:] = partial[-1]
    return partial


def _qite_exact(spec: Spectrum, tau: float) -> tuple[float, float]:
    r"""Return :math:`(e, g)` of the untruncated filter: off the target, and along it."""
    t = spec.target
    lam = 2.0 * (spec.singular / spec.alpha_r) ** 2  # 1 + x_j
    f = np.exp(-tau * lam)
    w = spec.weights
    gamma2 = float(np.sum(w[:t]))
    along = float(f[:t] @ w[:t]) / gamma2
    off = float(((f[:t] - along) ** 2) @ w[:t] + (f[t:] ** 2) @ w[t:])
    return math.sqrt(off), along * math.sqrt(gamma2)


def _qite_floor(spec: Spectrum, eps: float) -> float:
    r"""Return the least :math:`\tau` at which the untruncated filter reaches ``eps``."""

    def ratio(tau: float) -> float:
        e, g = _qite_exact(spec, tau)
        return e / g if g > 0.0 else math.inf

    target = math.sqrt(eps)
    gap = float(2.0 * (spec.singular[spec.target] ** 2 - spec.singular[spec.target - 1] ** 2))
    high = spec.alpha_r**2 / gap
    while ratio(high) > target:
        high *= 2.0
    low = 0.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if ratio(middle) > target:
            low = middle
        else:
            high = middle
        if high - low <= 1e-12 * high:
            break
    return high


def _qite_state(
    spec: Spectrum, eps: float, max_degree: int
) -> tuple[tuple[float, int, int, float, float] | None, tuple[float, int] | None]:
    r"""Return QITE's best measured choice and best certified choice over :math:`\tau`.

    The measured choice is ``(tau, degree, certified degree at tau, infidelity, probability)``;
    the certified one ``(tau, degree)``.
    """
    s = math.sqrt(eps)

    def certified(tau: float) -> int | None:
        e, g = _qite_exact(spec, tau)
        if g * g < RESOLVABLE_PROBABILITY:
            return None
        return _certified_degree(tau, e, g, s, max_degree)

    certified_best = _search(_qite_floor(spec, eps), certified)

    cache: dict[float, tuple[int, int, float, float] | None] = {}

    def measured(tau: float) -> int | None:
        if tau not in cache:
            top = certified(tau)
            if top is None:
                cache[tau] = None
            else:
                e, g = _qite_exact(spec, tau)
                tolerance = (s * g - e) / (1.0 + s)
                coefficients = qite_coefficients(tau, tolerance=tolerance, max_degree=max_degree)
                c = coefficients.coefficients
                running = np.zeros(spec.singular.size)

                def infidelities(start: int, stop: int) -> NDArray[np.float64]:
                    return spec.infidelity(_qite_values(spec, c, start, stop, running))

                last = _last_violation(infidelities, top, eps)
                if last >= top:
                    raise RuntimeError(f"QITE's certified degree {top} misses {eps} at tau={tau}")
                degree = last + 1
                values = _qite_partial(spec, c, degree)
                cache[tau] = (
                    degree,
                    top,
                    float(spec.infidelity(values[None, :])[0]),
                    float(spec.probability(values[None, :])[0]),
                )
        entry = cache[tau]
        return None if entry is None else entry[0]

    measured_best = _search(_qite_floor(spec, eps), measured)
    if measured_best is None:
        return None, certified_best
    tau, degree = measured_best
    entry = cache[tau]
    assert entry is not None
    return (tau, degree, entry[1], entry[2], entry[3]), certified_best


def _qite_partial(
    spec: Spectrum, coefficients: NDArray[np.float64], degree: int
) -> NDArray[np.float64]:
    """Return the degree-``degree`` partial sum at the eigenvalues."""
    running = np.zeros(spec.singular.size)
    for start in range(0, degree + 1, _CHUNK):
        stop = min(start + _CHUNK, degree + 1)
        _qite_values(spec, coefficients, start, stop, running)
    return running


@dataclass(frozen=True)
class StateComparison:
    r"""QITE and the minimax filter at one infidelity on one spectrum and initial state.

    Degrees are ``None`` where the filter cannot reach :math:`\varepsilon` within the ceiling.

    Attributes
    ----------
    epsilon
        The infidelity :math:`\varepsilon` both are held to.
    minimax_certified, minimax_degree
        The minimax filter's least degree from the worst-case bound at :math:`\gamma`, and its
        least measured degree (every larger degree up to the certified one also reaches
        :math:`\varepsilon`).
    minimax_infidelity, minimax_probability
        At the measured degree.
    qite_worst
        QITE's least certified degree from the same worst-case bound, over :math:`\tau`.
    qite_certified, qite_certified_tau
        QITE's least degree certified on the actual spectrum and state, over :math:`\tau`.
    qite_degree, qite_tau, qite_infidelity, qite_probability
        QITE's least measured degree over :math:`\tau`, with the same rule as the minimax
        filter's.
    production_degree, production_tau, production_infidelity, production_probability
        QITE as the pipeline builds it for :math:`\varepsilon_f=\varepsilon`
        (:func:`~pihm.qsp.filters.qite_filter`), measured at its degree.
    minimax_ground, qite_ground
        :math:`|P(x_0)|` for each at its measured degree.
    """

    epsilon: float
    minimax_certified: int
    minimax_degree: int
    minimax_infidelity: float
    minimax_probability: float
    minimax_ground: float
    qite_worst: int | None
    qite_certified: int | None
    qite_certified_tau: float | None
    qite_degree: int | None
    qite_tau: float | None
    qite_infidelity: float | None
    qite_probability: float | None
    qite_ground: float | None
    production_degree: int | None
    production_tau: float
    production_infidelity: float | None
    production_probability: float | None


def compare_state(
    spec: Spectrum, epsilon: float, *, max_degree: int = MAX_DEGREE
) -> StateComparison:
    """Compare QITE and the minimax filter at equal infidelity on a spectrum (module docstring)."""
    certified, degree, infidelity = _minimax_state(spec, epsilon)
    ground = _minimax_ground(spec, degree)
    ratios = _minimax_ratios(spec, np.array([degree]))[0] * ground
    probability = float(spec.probability(ratios[None, :])[0])

    gamma2 = spec.gamma**2
    eta = math.sqrt(epsilon * gamma2 / (1.0 - gamma2)) if gamma2 < 1.0 else 0.5
    worst = qite_suppression_degree(
        spec.x_ground, spec.x_excited, min(eta, 0.5), max_degree=max_degree
    )
    measured, certified_best = _qite_state(spec, epsilon, max_degree)

    delta = spec.x_excited - spec.x_ground
    production_tau = math.log(1.0 / epsilon) / delta
    production_degree: int | None = None
    production_infidelity: float | None = None
    production_probability: float | None = None
    try:
        built = qite_coefficients(production_tau, tolerance=epsilon / 10.0, max_degree=max_degree)
    except ValueError:
        pass
    else:
        production_degree = built.degree
        values = _qite_partial(spec, built.coefficients, built.degree)
        production_infidelity = float(spec.infidelity(values[None, :])[0])
        production_probability = float(spec.probability(values[None, :])[0])

    return StateComparison(
        epsilon=epsilon,
        minimax_certified=certified,
        minimax_degree=degree,
        minimax_infidelity=infidelity,
        minimax_probability=probability,
        minimax_ground=ground,
        qite_worst=None if worst is None else worst.degree,
        qite_certified=None if certified_best is None else certified_best[1],
        qite_certified_tau=None if certified_best is None else certified_best[0],
        qite_degree=None if measured is None else measured[1],
        qite_tau=None if measured is None else measured[0],
        qite_infidelity=None if measured is None else measured[3],
        qite_probability=None if measured is None else measured[4],
        qite_ground=None if measured is None else math.exp(-measured[0] * (1.0 + spec.x_ground)),
        production_degree=production_degree,
        production_tau=production_tau,
        production_infidelity=production_infidelity,
        production_probability=production_probability,
    )


def qite_state_coefficients(
    spec: Spectrum, epsilon: float, tau: float, degree: int, *, max_degree: int = MAX_DEGREE
) -> NDArray[np.float64]:
    """Return the QITE coefficients :func:`compare_state` measured, truncated at ``degree``.

    They are the certified series at ``tau`` for ``epsilon`` on this spectrum, cut at the measured
    degree, for applying the chosen filter to a state or finding its phases.
    """
    e, g = _qite_exact(spec, tau)
    s = math.sqrt(epsilon)
    full = qite_coefficients(tau, tolerance=(s * g - e) / (1.0 + s), max_degree=max_degree)
    return np.asarray(full.coefficients[: degree + 1], dtype=np.float64)
