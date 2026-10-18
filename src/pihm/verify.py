r"""Verify a block encoding against an independently assembled operator.

Three modes, all comparing :class:`~pihm.circuits.primitives.BlockEncoding` against a target
built independently (from :mod:`pihm.operators`, never from the circuit itself):

* ``"exact"``: the full top-left block, via :func:`~pihm.circuits.primitives.top_left_block`
  (one statevector evolution per system basis state). Feasible only up to a modest total qubit
  count (the design's own ceiling is about 20).
* ``"probe"``: a handful of Haar-random and structured vectors, via
  :func:`~pihm.circuits.primitives.apply`, each one statevector evolution regardless of system
  size. Feasible well past exact mode's ceiling.
* ``"approximation"``: for an encoder that is not exact by construction (the published
  :math:`\mathbb{G}`), the measured error is reported against a caller-supplied theoretical
  bound, and does not gate a pass/fail verdict on its own ("report, don't gate").

The target is accepted as a dense array or a SciPy sparse array; sparse targets are only ever
multiplied against a probe vector (never densified), so probe mode scales to the sizes exact mode
cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm.circuits.primitives import BlockEncoding, apply, compile_circuit, top_left_block
from pihm.operators.tensor import SparseArray

__all__ = ["VerificationResult", "verify_approximation", "verify_exact", "verify_probe"]

#: Structured probe vectors, added to the Haar-random ones in :func:`verify_probe`.
_STRUCTURED_LABELS = ("e_first", "e_last", "e_middle", "uniform")


@dataclass(frozen=True)
class VerificationResult:
    r"""The outcome of one verification run.

    Attributes
    ----------
    mode
        Which of the three modes produced this result.
    max_error
        The largest absolute error found (entrywise for ``"exact"``, over probe vectors for
        ``"probe"``, the single measured value for ``"approximation"``).
    tolerance
        The threshold ``max_error`` was compared against (``None`` for ``"approximation"``,
        which reports rather than gates).
    passed
        Whether ``max_error`` met ``tolerance`` (always ``True`` for ``"approximation"``).
    details
        Mode-specific extras: e.g. ``{"probe_errors": {...}}`` or
        ``{"bound": ..., "within_bound": ...}``.
    """

    mode: Literal["exact", "probe", "approximation"]
    max_error: float
    tolerance: float | None
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def _to_dense(target: SparseArray | NDArray[Any]) -> NDArray[np.complexfloating]:
    """Return ``target`` as a dense complex array."""
    return np.asarray(target.todense() if sp.issparse(target) else target, dtype=complex)


def verify_exact(
    encoding: BlockEncoding, target: SparseArray | NDArray[Any], *, atol: float | None = None
) -> VerificationResult:
    r"""Compare the full reconstructed block against ``target``.

    Parameters
    ----------
    encoding
        The block encoding under test.
    target
        The independently assembled operator, shape ``encoding.shape``.
    atol
        Absolute tolerance; defaults to :math:`10^{-10} \|{\rm target}\|_2`,
        or :math:`10^{-10}` if the target is identically zero.

    Returns
    -------
    VerificationResult
        ``max_error`` is the largest entrywise absolute error.

    Raises
    ------
    ValueError
        If ``target``'s shape does not match ``encoding.shape``.
    """
    dense = _to_dense(target)
    if dense.shape != encoding.shape:
        raise ValueError(f"target shape {dense.shape} != encoding.shape {encoding.shape}")
    if atol is None:
        norm = float(np.linalg.norm(dense, 2)) if dense.size else 0.0
        atol = 1e-10 * norm if norm > 0 else 1e-10
    block = top_left_block(encoding)
    max_error = float(np.max(np.abs(block - dense))) if dense.size else 0.0
    return VerificationResult("exact", max_error, atol, max_error <= atol)


def _structured_probes(
    cols: int, rng: np.random.Generator
) -> dict[str, NDArray[np.complexfloating]]:
    """Return the fixed structured probe vectors (block boundaries and a uniform state)."""
    e_first = np.zeros(cols, dtype=complex)
    e_first[0] = 1.0
    e_last = np.zeros(cols, dtype=complex)
    e_last[-1] = 1.0
    e_middle = np.zeros(cols, dtype=complex)
    e_middle[cols // 2] = 1.0
    uniform = np.full(cols, 1.0 / np.sqrt(cols), dtype=complex)
    return dict(zip(_STRUCTURED_LABELS, (e_first, e_last, e_middle, uniform), strict=True))


def verify_probe(
    encoding: BlockEncoding,
    target: SparseArray | NDArray[Any],
    rng: np.random.Generator,
    *,
    n_random: int = 16,
    rtol: float = 1e-9,
) -> VerificationResult:
    r"""Compare :func:`~pihm.circuits.primitives.apply` against sparse matrix-vector products.

    Parameters
    ----------
    encoding
        The block encoding under test.
    target
        The independently assembled operator (dense or sparse), shape ``encoding.shape``.
    rng
        Source of the Haar-random probe vectors.
    n_random
        How many random probes to add to the four structured ones (16-32
        total).
    rtol
        Relative-error tolerance, against ``max(1, ||target @ probe||)``.

    Returns
    -------
    VerificationResult
        ``max_error`` is the largest relative error over every probe; ``details["probe_errors"]``
        names each one.

    Raises
    ------
    ValueError
        If ``target``'s shape does not match ``encoding.shape``.
    """
    rows, cols = encoding.shape
    if target.shape != (rows, cols):
        raise ValueError(f"target shape {target.shape} != encoding.shape {encoding.shape}")
    probes = dict(_structured_probes(cols, rng))
    for i in range(n_random):
        v = rng.normal(size=cols) + 1j * rng.normal(size=cols)
        probes[f"random_{i}"] = v / np.linalg.norm(v)

    compiled = compile_circuit(encoding.circuit)
    errors: dict[str, float] = {}
    for name, v in probes.items():
        got = apply(encoding, v, compiled)
        expected = np.asarray(target @ v).reshape(-1)
        scale = max(1.0, float(np.linalg.norm(expected)))
        errors[name] = float(np.linalg.norm(got - expected) / scale)
    max_error = max(errors.values()) if errors else 0.0
    return VerificationResult("probe", max_error, rtol, max_error <= rtol, {"probe_errors": errors})


def verify_approximation(
    encoding: BlockEncoding,
    target: SparseArray | NDArray[Any],
    bound: float,
) -> VerificationResult:
    r"""Measure an inexact encoder's entrywise error against a caller-supplied bound.

    Does not gate: ``passed`` is always ``True``, matching the "report, don't
    gate" for the published :math:`\mathbb{G}`'s approximation mode. Only feasible at exact
    mode's own scale, since it needs the full reconstructed block.

    Parameters
    ----------
    encoding
        The (inexact) block encoding under test.
    target
        The exact reference operator, shape ``encoding.shape``.
    bound
        The theoretical error bound to report alongside the measured value (for example
        :math:`\alpha \cdot` :func:`pihm.circuits.standard.derivative_ladder_error_bound`).

    Returns
    -------
    VerificationResult
        ``max_error`` is the largest entrywise absolute error; ``details`` names ``bound`` and
        whether the measurement fell within it.
    """
    dense = _to_dense(target)
    block = top_left_block(encoding)
    max_error = float(np.max(np.abs(block - dense))) if dense.size else 0.0
    return VerificationResult(
        "approximation", max_error, None, True, {"bound": bound, "within_bound": max_error <= bound}
    )
