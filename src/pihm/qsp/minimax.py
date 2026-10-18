r"""The minimax edge filter, the alternative to imaginary time.

The filter acts on :math:`H'=2H/\alpha_R^2-1`, whose spectrum lies in :math:`[-1,1]` and, since
:math:`H\ge0`, reaches down to :math:`-1` at most. To keep the eigenvalues below :math:`x_1` (the
ground state, :math:`x_0<x_1`) and suppress :math:`[x_1,1]`, it is the Chebyshev polynomial of
least deviation on the suppressed interval, anchored at the spectrum's lower bound:

.. math::

   p_d(x) = \frac{T_d(w(x))}{T_d(w(-1))},\qquad w(x)=\frac{2x-(x_1+1)}{1-x_1},

with :math:`w` mapping :math:`[x_1,1]` onto :math:`[-1,1]`, where :math:`|T_d|\le1`. Below
:math:`x_1`, :math:`|T_d(w)|` grows monotonically towards :math:`x=-1`, so :math:`|p_d|\le1` on
the whole of :math:`[-1,1]` (GQSP's condition) with equality at :math:`-1`. Relative to the ground
state it suppresses the excited spectrum by

.. math::

   \eta = \sup_{x\in[x_1,1]}\frac{|p_d(x)|}{|p_d(x_0)|} = \frac1{\cosh(d\,\theta_0)},\qquad
   \theta_0=\operatorname{arccosh}(-w(x_0)),

which no polynomial of degree :math:`d` bounded by 1 at :math:`x_0` beats on :math:`[x_1,1]`
(Chebyshev's theorem on least deviation). It is the eigenstate-filtering polynomial of Lin and
Tong, *Quantum* **4**, 372 (2020), at the spectrum's edge. Values are formed from the closed
form, with the cosh ratios taken in logarithms: :math:`T_d(w)` for :math:`w<-1` is
:math:`(-1)^d\cosh(d\,\operatorname{arccosh}(-w))`, whose sign cancels against the anchor's, while
on :math:`[-1,1]` it is :math:`\cos(d\arccos w)` and keeps the sign :math:`(-1)^d` the anchor
divides in.

**The pipeline's rule**: state preparation uses this filter
at the least degree whose suppression is :math:`\varepsilon`, :func:`edge_degree`. That is the
guarantee QITE's rule gives (:func:`pihm.qsp.filters.qite_filter`), held relative to the ground
state rather than by an absolute tail. Its inputs are heights above the spectrum's edge,
:math:`h=1+x=2s^2/\alpha_R^2`, which keep their relative precision where :math:`x` itself would
round to :math:`-1` (:func:`edge_angles`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dct

from pihm.qsp.filters import MAX_DEGREE

__all__ = [
    "MinimaxFilter",
    "edge_angles",
    "edge_degree",
    "edge_suppression",
    "minimax_degree",
    "minimax_filter",
    "minimax_suppression",
    "minimax_values",
]


def _w(x: ArrayLike, x_excited: float) -> NDArray[np.float64]:
    """Return the affine map taking ``[x_excited, 1]`` onto ``[-1, 1]``."""
    return (2.0 * np.asarray(x, dtype=np.float64) - (x_excited + 1.0)) / (1.0 - x_excited)


def _cosh_ratio(u: NDArray[np.float64], v: float) -> NDArray[np.float64]:
    """Return ``cosh(u) / cosh(v)`` for ``u, v >= 0`` without forming either cosh."""
    ratio: NDArray[np.float64] = np.exp(u - v) * (1.0 + np.exp(-2.0 * u)) / (1.0 + np.exp(-2.0 * v))
    return ratio


def _check(x_excited: float) -> None:
    if not -1.0 < x_excited < 1.0:
        raise ValueError(f"the suppressed interval must start inside (-1, 1), got {x_excited}")


def minimax_values(x: ArrayLike, degree: int, x_excited: float) -> NDArray[np.float64]:
    r"""Return :math:`p_d(x)` exactly, from the closed form (module docstring).

    Parameters
    ----------
    x
        Points in :math:`[-1,1]`.
    degree
        :math:`d\ge0`.
    x_excited
        :math:`x_1`, where the suppressed interval starts.
    """
    _check(x_excited)
    w = _w(x, x_excited)
    anchor = float(degree * np.arccosh(-_w(-1.0, x_excited)))
    out = np.empty_like(w)
    below = w < -1.0
    out[below] = _cosh_ratio(degree * np.arccosh(-w[below]), anchor)
    inside = ~below
    sign = -1.0 if degree % 2 else 1.0
    scale = float(_cosh_ratio(np.zeros(1), anchor)[0])
    out[inside] = sign * np.cos(degree * np.arccos(np.clip(w[inside], -1.0, 1.0))) * scale
    return out


def minimax_suppression(degree: int, x_ground: float, x_excited: float) -> float:
    r"""Return :math:`\eta=1/\cosh(d\,\theta_0)`, the excited spectrum relative to the ground."""
    _check(x_excited)
    theta = float(np.arccosh(-_w(x_ground, x_excited)))
    return float(_cosh_ratio(np.zeros(1), degree * theta)[0])


def minimax_degree(x_ground: float, x_excited: float, suppression: float) -> int:
    r"""Return the least degree with :math:`\eta\le` ``suppression``.

    That is :math:`\lceil\operatorname{arccosh}(1/\eta)/\theta_0\rceil`, with :math:`\eta` the
    ``suppression`` and :math:`\theta_0` as in the module docstring.
    """
    _check(x_excited)
    if not -1.0 <= x_ground < x_excited:
        raise ValueError(f"the ground state must lie below the suppressed interval, got {x_ground}")
    if not 0.0 < suppression < 1.0:
        raise ValueError(f"the suppression must lie in (0, 1), got {suppression}")
    theta = float(np.arccosh(-_w(x_ground, x_excited)))
    return math.ceil(float(np.arccosh(1.0 / suppression)) / theta)


def _arccosh1p(e: ArrayLike) -> NDArray[np.float64]:
    r"""Return :math:`\operatorname{arccosh}(1+e)` for :math:`e\ge0`, never forming :math:`1+e`."""
    e = np.asarray(e, dtype=np.float64)
    result: NDArray[np.float64] = np.log1p(e + np.sqrt(e * (2.0 + e)))
    return result


def edge_angles(heights: ArrayLike, excited: float) -> NDArray[np.float64]:
    r"""Return :math:`\theta=\operatorname{arccosh}(-w(x))` below the suppressed interval.

    Each point is given by its height :math:`h=1+x` above the spectrum's edge, and ``excited`` is
    :math:`h_1=1+x_1`. Since :math:`-w(x)-1=2(h_1-h)/(2-h_1)`, the angle keeps its relative
    precision where the spectrum crowds against :math:`-1`.
    """
    h = np.asarray(heights, dtype=np.float64)
    return _arccosh1p(2.0 * (excited - h) / (2.0 - excited))


def edge_degree(ground: float, excited: float, suppression: float) -> int:
    r"""Return the least degree that suppresses :math:`[x_1,1]` to ``suppression``, from heights.

    :func:`minimax_degree`, :math:`\lceil\operatorname{arccosh}(1/\eta)/\theta_0\rceil`, with
    the ground and excited values given as heights :math:`1+x_0` and :math:`1+x_1`
    (:func:`edge_angles`): the pipeline's rule for the minimax filter (the module docstring).

    Raises
    ------
    ValueError
        Unless :math:`0\le1+x_0<1+x_1<2` and :math:`0<` ``suppression`` :math:`<1`.
    """
    if not 0.0 <= ground < excited < 2.0:
        raise ValueError(f"need 0 <= ground < excited < 2 as heights, got {ground}, {excited}")
    if not 0.0 < suppression < 1.0:
        raise ValueError(f"the suppression must lie in (0, 1), got {suppression}")
    theta = float(edge_angles(ground, excited))
    return math.ceil(float(np.arccosh(1.0 / suppression)) / theta)


def edge_suppression(heights: ArrayLike, degree: int, excited: float) -> NDArray[np.float64]:
    r"""Return :math:`|p_d(x_1)|/|p_d(x)|=1/\cosh(d\,\theta)` at points below :math:`x_1`.

    The suppression of :math:`[x_1,1]` relative to each point, given by heights as in
    :func:`edge_angles`: at the ground state, the :math:`\eta` the filter achieves.
    """
    angles = degree * edge_angles(heights, excited)
    result: NDArray[np.float64] = 2.0 * np.exp(-angles) / (1.0 + np.exp(-2.0 * angles))
    return result


@dataclass(frozen=True)
class MinimaxFilter:
    r"""The minimax edge filter's Chebyshev coefficients (module docstring).

    Attributes
    ----------
    x_excited
        :math:`x_1`, where the suppressed interval starts.
    coefficients
        :math:`p_0,\dots,p_d`, low-to-high degree, the :math:`T_k` basis on :math:`[-1,1]`.
    """

    x_excited: float
    coefficients: NDArray[np.float64]

    @property
    def degree(self) -> int:
        """The degree :math:`d`."""
        return int(self.coefficients.shape[0] - 1)

    @property
    def weight(self) -> float:
        r""":math:`\sum_k|p_k|`, the filter's value at :math:`-1`: 1, up to rounding."""
        return float(np.sum(np.abs(self.coefficients)))

    @property
    def tail_bound(self) -> float:
        """Nothing is truncated, so zero, the coefficients being the filter itself."""
        return 0.0


def minimax_filter(degree: int, x_excited: float, max_degree: int = MAX_DEGREE) -> MinimaxFilter:
    r"""Return :math:`p_d`'s Chebyshev coefficients, by exact interpolation.

    :math:`p_d` is sampled at :math:`N>d` Chebyshev–Gauss nodes and read off by a type-II discrete
    cosine transform, which is exact for a polynomial of degree below :math:`N`: interpolation,
    not a fit.

    Raises
    ------
    ValueError
        If ``degree`` exceeds ``max_degree``.
    """
    _check(x_excited)
    if degree > max_degree:
        raise ValueError(f"degree {degree} exceeds max_degree={max_degree}")
    size = 1 << max(4, int(np.ceil(np.log2(2 * (degree + 1)))))
    nodes = np.cos(np.pi * (2 * np.arange(size) + 1) / (2 * size))
    coefficients = dct(minimax_values(nodes, degree, x_excited), type=2) / size
    coefficients[0] *= 0.5
    return MinimaxFilter(x_excited, np.asarray(coefficients[: degree + 1], dtype=np.float64))
