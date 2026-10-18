r"""The initial states a preparation tries.

Every state is answer-independent: it never reads the equation's own solution. The primary one is
the geometric product state, tried at each of
:data:`GEOMETRIC_R`'s values of :math:`r`, the one with the larger measured success probability
kept; the uniform and random states are its controls. :func:`initial_states` places the state on
a residual's field blocks, where the readout reads, and zero on every other block: the
descriptor's derivative blocks and padding, a Carleman lift's higher monomials. Every tier of
:mod:`pihm.qsp.prepare` starts from it.

**A system's fields** carry the state weighted by the problem's own initial data,
:math:`w_0/\|w_0\|`, the values its ratio constraint states.
The data are part of the problem's statement, never its solution, so the state stays
answer-independent. Equal weights, the first rule, depended on the sign each basis
function happens to carry, and cancelled exactly on the Taylor–Green vortex, whose two modes carry
opposite signs.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pihm.assemble import Residual
from pihm.problems.reference import slice_data

__all__ = [
    "GEOMETRIC_R",
    "InitialKind",
    "field_weights",
    "geometric_state",
    "initial_states",
    "random_state",
    "uniform_state",
]

InitialKind: TypeAlias = Literal["geometric", "uniform", "random"]

#: The r values of the geometric initial states tried, in order;
#: the one with the larger measured success probability is kept.
GEOMETRIC_R: Final[tuple[float, ...]] = (0.5, -0.5)


def geometric_state(dimension: int, r: float) -> NDArray[np.complex128]:
    r"""Return the normalised geometric product state :math:`\propto r^k`, :math:`k=0,\dots,d-1`.

    The primary initial state is the product state
    :math:`\bigotimes_j(|0\rangle+r^{2^j}|1\rangle)`, whose coefficients in the computational
    basis are exactly :math:`r^k`. Answer-independent (it never reads the equation's own
    solution).
    """
    if dimension < 1:
        raise ValueError(f"geometric_state needs dimension >= 1, got {dimension}")
    if r == 0.0:
        raise ValueError("geometric_state needs a nonzero r (r = 0 gives a rank-1, non-unit state)")
    powers = np.asarray(r, dtype=np.float64) ** np.arange(dimension, dtype=np.float64)
    result: NDArray[np.complex128] = (powers / np.linalg.norm(powers)).astype(np.complex128)
    return result


def uniform_state(dimension: int) -> NDArray[np.complex128]:
    """Return the normalised uniform state."""
    if dimension < 1:
        raise ValueError(f"uniform_state needs dimension >= 1, got {dimension}")
    return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)


def random_state(dimension: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Return a normalised complex Gaussian state."""
    if dimension < 1:
        raise ValueError(f"random_state needs dimension >= 1, got {dimension}")
    raw = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    result: NDArray[np.complex128] = (raw / np.linalg.norm(raw)).astype(np.complex128)
    return result


def _initial_candidates(
    kind: InitialKind,
    dimension: int,
    geometric_r: tuple[float, ...],
    rng: np.random.Generator | None,
) -> list[NDArray[np.complex128]]:
    """Return the kind's states on ``dimension`` amplitudes, before they are placed."""
    if kind == "geometric":
        if not geometric_r:
            raise ValueError("need at least one geometric_r when kind='geometric'")
        return [geometric_state(dimension, r) for r in geometric_r]
    if kind == "uniform":
        return [uniform_state(dimension)]
    if kind == "random":
        return [random_state(dimension, rng or np.random.default_rng())]
    raise ValueError(f"unknown kind {kind!r}, expected 'geometric', 'uniform' or 'random'")


def initial_states(
    kind: InitialKind,
    residual: Residual,
    geometric_r: tuple[float, ...] = GEOMETRIC_R,
    rng: np.random.Generator | None = None,
) -> list[NDArray[np.complex128]]:
    """Return the initial states to try for ``residual``, in ``geometric_r``'s order.

    Each is the kind's state on the field block and zero on every other block:
    the geometric state for each :math:`r`, or the one uniform or random state (``rng`` draws
    it, an unseeded generator by default). In the standard form of a single field the field block
    is the state. A system's fields each carry the state times their weight
    (:func:`field_weights`), and its other blocks (a Carleman lift's higher monomials, the
    padding) nothing.
    """
    size = residual.dimension // len(residual.derivatives)
    fields = residual.field_blocks
    weights = field_weights(residual)
    placed = []
    for candidate in _initial_candidates(kind, size, geometric_r, rng):
        state = np.zeros(residual.dimension, dtype=np.complex128)
        for field, weight in zip(fields, weights, strict=True):
            state[field * size : (field + 1) * size] = weight * candidate
        placed.append(state)
    return placed


def field_weights(residual: Residual) -> NDArray[np.float64]:
    r"""Return each field's weight in the initial state: 1 alone, or a system's :math:`w_0/\|w_0\|`.

    :math:`w_0` is the fields' values at the system's ratio constraint, the data the problem
    states; a field whose datum is zero starts empty.
    """
    if len(residual.field_blocks) == 1:
        return np.ones(1)
    (constraint,) = [c for c in residual.problem.constraints if c.kind == "ratio"]
    data = slice_data(residual.problem, constraint.axis, constraint.at)
    return np.asarray(data / np.linalg.norm(data), dtype=np.float64)
