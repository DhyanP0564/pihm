r"""Logical and transpiled resource counts, and scaling fits.

:func:`count_resources` reports a :class:`~pihm.circuits.primitives.BlockEncoding`'s qubits,
:math:`\alpha`, logical gate counts (:meth:`~qiskit.circuit.QuantumCircuit.count_ops`) and its
transpiled cost on the ``{cx, u}`` basis, at ``optimization_level=1`` (the design's own fixed
choice, so counts are comparable across encoders and sessions), from
:func:`~pihm.circuits.primitives.native_circuit`'s multi-controlled gates, dense gates left as
cited primitives. :func:`fit_power_law`
turns a handful of ``(n, value)`` points into the exponent a scaling claim
makes (for example "gates scale as :math:`\Theta(n^2)`").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from pihm.circuits.primitives import BlockEncoding, native_circuit

__all__ = [
    "BASIS_GATES",
    "OPTIMISATION_LEVEL",
    "ResourceCounts",
    "count_resources",
    "fit_power_law",
]

#: The fixed transpilation basis every reported count uses.
BASIS_GATES: tuple[str, ...] = ("cx", "u")

#: The fixed optimisation level every reported count uses.
OPTIMISATION_LEVEL = 1


@dataclass(frozen=True)
class ResourceCounts:
    r"""Resource counts of one block encoding.

    Attributes
    ----------
    n_sys_qubits, n_anc_qubits, n_qubits
        The register split and its total.
    alpha
        The subnormalisation.
    logical_gate_counts
        :meth:`~qiskit.circuit.QuantumCircuit.count_ops` on the untranspiled circuit: names such
        as ``"unitary"`` or ``"ry"`` map to how many instructions of that kind appear, without
        decomposing any of them further.
    transpiled_gate_counts
        The same, after :func:`~qiskit.compiler.transpile` onto :data:`BASIS_GATES` at
        :data:`OPTIMISATION_LEVEL`: every instruction reduced to controlled-NOT and a general
        single-qubit unitary. Dense gates are counted apart (``cited_count``): each is replaced
        by a barrier on its qubits before transpiling, so nothing is optimised across it.
    cx_count, one_qubit_count
        ``transpiled_gate_counts["cx"]`` and the sum of every other gate's entry (0 if absent).
    cited_count
        The dense gates left unsynthesised: the Chebyshev–Gauss transform of the doubled
        space's fold, and its controlled copies, which stand for the discrete cosine transform
        is budgeted as a cited :math:`O(n^2)` primitive, so a count
        reads "cited × cited_count + the rest". Zero for every other circuit.
    depth
        The transpiled circuit's depth, not counting the cited dense gates.
    """

    n_sys_qubits: int
    n_anc_qubits: int
    n_qubits: int
    alpha: float
    logical_gate_counts: Mapping[str, int]
    transpiled_gate_counts: Mapping[str, int]
    cx_count: int
    one_qubit_count: int
    cited_count: int
    depth: int


def count_resources(encoding: BlockEncoding) -> ResourceCounts:
    """Return :class:`ResourceCounts` for ``encoding``.

    Parameters
    ----------
    encoding
        The block encoding to measure.

    Returns
    -------
    ResourceCounts
    """
    logical = dict(encoding.circuit.count_ops())
    native = native_circuit(encoding.circuit)
    stripped = QuantumCircuit(*native.qregs, global_phase=native.global_phase)
    cited = 0
    for inst in native.data:
        if isinstance(inst.operation, UnitaryGate):
            stripped.barrier(*inst.qubits)
            cited += 1
        else:
            stripped.append(inst.operation, inst.qubits)
    transpiled = transpile(
        stripped, basis_gates=list(BASIS_GATES), optimization_level=OPTIMISATION_LEVEL
    )
    counts = dict(transpiled.count_ops())
    cx = counts.get("cx", 0)
    counts.pop("barrier", None)
    one_qubit = sum(count for name, count in counts.items() if name != "cx")
    return ResourceCounts(
        n_sys_qubits=encoding.n_sys_qubits,
        n_anc_qubits=encoding.n_anc_qubits,
        n_qubits=encoding.n_qubits,
        alpha=encoding.alpha,
        logical_gate_counts=logical,
        transpiled_gate_counts=counts,
        cx_count=cx,
        one_qubit_count=one_qubit,
        cited_count=cited,
        depth=transpiled.depth(),
    )


def fit_power_law(ns: Sequence[float], values: Sequence[float]) -> tuple[float, float]:
    r"""Fit ``values ~= coefficient * ns**exponent`` by a log-log least-squares line.

    Parameters
    ----------
    ns
        Independent variable (usually :math:`N = 2^n`, not the qubit count :math:`n` itself:
        scaling claims such as :math:`\Theta(N^2)` are stated in
        :math:`N`), strictly positive.
    values
        Measured quantities, strictly positive, same length as ``ns``.

    Returns
    -------
    coefficient, exponent
        So that ``values[i] ~= coefficient * ns[i]**exponent``.

    Raises
    ------
    ValueError
        If fewer than two points are given, or any value is non-positive.
    """
    x = np.asarray(ns, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("fit_power_law needs at least two points")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("fit_power_law needs strictly positive ns and values")
    exponent, log_c = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_c)), float(exponent)
