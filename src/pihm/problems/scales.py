r"""The scales that set the descriptor's block rescaling.

The descriptor form carries each derivative of the field as a block of its state, so its
Hamiltonian measures a state by every derivative it carries. Its gap then depends on how well the
equation controls each of them, and the derivative blocks crowd the field block out of the ground
state. Every descriptor residual therefore carries the block of order :math:`m` along axis
:math:`a` divided by :math:`\lambda_a^m`, and :func:`block_scales` returns the :math:`\lambda_a`,
from the equation and the resolution alone (never fitted to a solution or a spectrum):

* **one axis, a leading coefficient** :math:`a_k` **with no zero on** :math:`[-1, 1]`:
  :math:`\lambda = 2\rho` with :math:`\rho = (\bar a_j/\bar a_k)^{1/(k-j)}` for the lowest order
  :math:`j < k` present, where :math:`\bar a_j` is the mean of :math:`|a_j|` over
  :math:`[-1, 1]`. For :math:`j = 0` and constant coefficients, :math:`\rho` is the geometric mean
  of the characteristic roots' magnitudes;
* **one axis, a leading coefficient that vanishes on** :math:`[-1, 1]`: :math:`\lambda = N/2`;
* **several axes:** :math:`\lambda_a = N_a`, times :math:`\sqrt{\max_b c_b / c_a}` on an axis
  whose top order is two, :math:`c_a` the mean magnitude of its top-order coefficient, so that
  those terms balance. For the wave :math:`f_{tt} = c^2 f_{xx}` this gives :math:`(cN, N)`.


"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.polynomial import legendre, polynomial

from pihm.problems.spec import Problem

__all__ = ["ROOT_TOLERANCE", "block_scales", "vanishes"]

#: The tolerance within which a root of the leading coefficient counts as real and on
#: :math:`[-1, 1]`. A double root, such as :math:`(1 - x^2)^2`'s, is computed only to about the
#: square root of machine precision, so the tolerance is loose.
ROOT_TOLERANCE: Final[float] = 1.0e-6

#: Gauss–Legendre points for the mean of a coefficient's magnitude over :math:`[-1, 1]`.
_MEAN_POINTS: Final[int] = 64


def vanishes(coefficients: Sequence[float]) -> bool:
    """Return whether a polynomial, by monomial coefficients, has a real zero on [-1, 1].

    Examples
    --------
    >>> vanishes([1.0, 0.0, -1.0]), vanishes([2.0, 1.0]), vanishes([1.0, 0.0, -2.0, 0.0, 1.0])
    (True, False, True)
    """
    c = np.trim_zeros(np.asarray(coefficients, dtype=np.float64), "b")
    if c.size == 0:
        return True
    if c.size == 1:
        return False
    roots = polynomial.polyroots(c)
    real = roots[np.abs(roots.imag) <= ROOT_TOLERANCE].real
    return bool(np.any(np.abs(real) <= 1.0 + ROOT_TOLERANCE))


def _mean_magnitude(coefficients: Sequence[float]) -> float:
    """Return the mean of a polynomial's magnitude over [-1, 1] (Gauss–Legendre)."""
    nodes, weights = legendre.leggauss(_MEAN_POINTS)
    return float(np.sum(weights * np.abs(polynomial.polyval(nodes, coefficients))) / 2.0)


def _coefficients(problem: Problem) -> dict[tuple[int, ...], list[float]]:
    """Return, per derivative, the monomial coefficients of the equation's multiplier of it."""
    (equation,) = problem.equations
    found: dict[tuple[int, ...], list[float]] = {}
    for term in equation.terms:
        if not term.factors:
            continue
        (factor,) = term.factors
        multiplier = term.multiplier.coefficients if term.multiplier else (1.0,)
        scaled = [term.coefficient * c for c in multiplier]
        known = found.get(factor.derivative, [])
        size = max(len(known), len(scaled))
        found[factor.derivative] = [
            (known[i] if i < len(known) else 0.0) + (scaled[i] if i < len(scaled) else 0.0)
            for i in range(size)
        ]
    return found


def block_scales(problem: Problem, ns: Sequence[int]) -> tuple[float, ...]:
    r"""Return the descriptor's block-rescaling scale :math:`\lambda_a` for each axis.

    Parameters
    ----------
    problem
        A linear problem with one field and no mixed derivatives.
    ns
        Qubits per axis of the field.

    Returns
    -------
    tuple of float
        One :math:`\lambda_a` per axis, by the rule of the module documentation.

    Examples
    --------
    >>> from pihm.problems import ladder
    >>> block_scales(ladder.problem("R2c"), [5])  # f'' + 5f' + 400f: 2 sqrt(400)
    (40.0,)
    >>> block_scales(ladder.problem("R3a"), [5])  # (1 - x^2) f'' ...: N/2
    (16.0,)
    >>> block_scales(ladder.problem("R5c"), [3, 3])  # f_tt = 4 f_xx: (c N, N)
    (16.0, 8.0)
    """
    found = _coefficients(problem)
    sizes = [1 << int(n) for n in ns]
    if len(sizes) == 1:
        orders = {derivative[0]: c for derivative, c in found.items()}
        k = max(orders)
        if vanishes(orders[k]):
            return (sizes[0] / 2.0,)
        top = _mean_magnitude(orders[k])
        for j in range(k):
            if j in orders and _mean_magnitude(orders[j]) > 0.0:
                return (2.0 * (_mean_magnitude(orders[j]) / top) ** (1.0 / (k - j)),)
        return (2.0,)
    top_order = [0] * len(sizes)
    top_size = [0.0] * len(sizes)
    for derivative, c in found.items():
        axis = int(np.argmax(derivative))
        if derivative[axis] > top_order[axis]:
            top_order[axis], top_size[axis] = derivative[axis], _mean_magnitude(c)
    second = [size for order, size in zip(top_order, top_size, strict=True) if order == 2]
    return tuple(
        float(n_a * (np.sqrt(max(second) / size) if order == 2 else 1.0))
        for n_a, order, size in zip(sizes, top_order, top_size, strict=True)
    )
