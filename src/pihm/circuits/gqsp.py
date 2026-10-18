r"""GQSP on the qubitised walk: the circuit, and its simulation one controlled walk at a time.

GQSP (Motlagh & Wiebe, arXiv:2308.01501) realises a polynomial in the
eigenvalues of a qubitised walk :math:`W` with the circuit :math:`R_d\,[C_0W]\,R_{d-1}\cdots
[C_0W]\,R_0`, times the global phase :math:`e^{i\lambda}`, for the phase triple
:math:`(\theta,\phi,\lambda)` that :func:`pihm.qsp.phases.find_phases` returns. :math:`C_0W` is
:math:`W` applied where a new *signal* qubit is :math:`|0\rangle`, and :math:`R_k` rotates the
signal qubit alone. The branch where the signal and the walk's ancilla are all zero (the leading
block, in this package's :class:`~pihm.circuits.primitives.BlockEncoding` convention) realises the
target polynomial in :math:`H'`, as :func:`pihm.qsp.phases.realised_polynomial` computes it one
eigenvalue at a time.

:math:`C_0W` is :func:`pihm.circuits.compose.controlled_walk`: only the walk's two reflections
carry the control, never :math:`U_R`'s own gates.

* :func:`gqsp_circuit` assembles the whole circuit as a block encoding, :math:`d` copies of the
  controlled walk: a circuit to inspect, count or check at small degree.
* :func:`evolve_gqsp` applies the same circuit to a statevector one controlled walk at a time (one
  :func:`~pihm.circuits.primitives.evolve` per step, the controlled walk compiled once), so its
  memory is the statevector's whatever the degree. Tier T1 and the readout simulate
  GQSP this way, and :func:`apply_gqsp` reads the leading block out as
  :func:`~pihm.circuits.primitives.apply` does for any block encoding.
* With ``selector=True``, :func:`evolve_gqsp` applies the whole circuit only where one more qubit,
  above the signal, is :math:`|1\rangle`: the controlled state preparation of the readout's LCU
  combiner. The rotations are then controlled on the selector, the walk's two
  reflections on the selector and the signal together, and :math:`\lambda` is a phase gate on the
  selector, since a controlled global phase is a relative one.
* A long :func:`evolve_gqsp` logs its progress every :data:`PROGRESS_SECONDS`: the step reached,
  the mean time a step, and the time left at that rate. A simulation of hours otherwise shows
  nothing between its start and its end, and a slow node only at the job's time limit.

The rotation matrix and the :math:`A(z)=\mathrm{diag}(z,1)` control convention are pinned to
``pihm.qsp.phases._column``'s own recursion by direct numerical construction: with :math:`R_k=
\begin{pmatrix}e^{i\phi_k}\cos\theta_k&e^{i\phi_k}\sin\theta_k\\\sin\theta_k&-\cos\theta_k
\end{pmatrix}`, the top entry of :math:`R_d\,A(z)\,R_{d-1}\cdots A(z)\,R_0\binom10` equals
``_column(theta, phi, [z])`` to machine precision for every :math:`(\theta,\phi,z)` checked.
:math:`\lambda` enters no individual rotation, matching how
:func:`~pihm.qsp.phases.realised_polynomial` applies it once, at the end.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from pihm.circuits.compose import controlled_walk
from pihm.circuits.primitives import (
    BlockEncoding,
    compile_circuit,
    controlled_matrix,
    evolve,
    unitary_gate,
)

__all__ = ["PROGRESS_SECONDS", "apply_gqsp", "evolve_gqsp", "gqsp_circuit", "rotation_gate"]

#: Wall time between :func:`evolve_gqsp`'s progress lines; a shorter simulation logs none.
PROGRESS_SECONDS: Final[float] = 600.0

logger = logging.getLogger(__name__)


def rotation_gate(theta: float, phi: float) -> NDArray[np.complex128]:
    r"""Return the GQSP rotation matrix :math:`R(\theta,\phi)`.

    :math:`\lambda=0`; see the module docstring for why :math:`\lambda` is never folded in here.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * phi)
    matrix = np.array([[phase * cos_t, phase * sin_t], [sin_t, -cos_t]], dtype=np.complex128)
    return matrix


def _degree(theta: NDArray[np.floating], phi: NDArray[np.floating]) -> int:
    """Return the degree a phase pair realises, checking the pair's shapes agree."""
    if theta.shape != phi.shape:
        raise ValueError(
            f"theta and phi must have the same shape, got {theta.shape} and {phi.shape}"
        )
    return int(theta.shape[0]) - 1


def _system_qubits(stacked: BlockEncoding) -> int:
    r"""Return the true system register's width: :math:`\log_2` of ``stacked``'s column count."""
    N = stacked.shape[1]
    n = N.bit_length() - 1
    if (1 << n) != N:
        raise ValueError(f"stacked.shape[1]={N} is not a power of two")
    return n


def gqsp_circuit(
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
    lam: float,
    stacked: BlockEncoding,
    *,
    rescale: float = 1.0,
    label: str = "",
) -> BlockEncoding:
    r"""Return the GQSP circuit realising ``theta``/``phi``/``lam`` in the walk's eigenvalues.

    Parameters
    ----------
    theta, phi, lam
        A phase triple, e.g. :class:`pihm.qsp.phases.Phases`'s own ``theta``/``phi``/``lam``.
    stacked
        :math:`U_R` (:func:`pihm.circuits.compose.stack_rows`), from which the controlled walk is
        built (:func:`~pihm.circuits.compose.controlled_walk`).
    rescale
        :attr:`pihm.qsp.phases.Phases.rescale`. :func:`pihm.qsp.phases.spectral_complement`
        (which found ``theta``/``phi``/``lam``) rescales its target polynomial to sup-norm
        :math:`1-\text{margin}` before finding phases for it, so the circuit's leading block is
        the *target* polynomial divided by this factor. Passing it here folds the correction into
        :attr:`~pihm.circuits.primitives.BlockEncoding.alpha`, so
        :func:`~pihm.circuits.primitives.apply` and
        :func:`~pihm.circuits.primitives.top_left_block` read out the true polynomial directly.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        The walk's system register and shape; its ancilla (row select and :math:`U_R`'s own)
        plus the signal qubit, the highest; :math:`\alpha=\text{rescale}`. Its top-left block,
        read via :func:`~pihm.circuits.primitives.apply` (which multiplies by :math:`\alpha`), is
        the target polynomial in :math:`H'`, to the accuracy
        :meth:`pihm.qsp.phases.Phases.max_error` certifies.
    """
    degree = _degree(theta, phi)
    n_sys = _system_qubits(stacked)
    n_walk = stacked.n_qubits
    signal = n_walk
    qc = QuantumCircuit(n_walk + 1, name=label or "GQSP")
    qc.append(unitary_gate(rotation_gate(theta[0], phi[0]), label="R0"), [signal])
    if degree >= 1:
        step = controlled_walk(stacked)
        for k in range(1, degree + 1):
            qc.compose(step, inplace=True)
            qc.append(unitary_gate(rotation_gate(theta[k], phi[k]), label=f"R{k}"), [signal])
    qc.global_phase += float(lam)
    N = 1 << n_sys
    return BlockEncoding(qc, float(rescale), n_sys, n_walk + 1 - n_sys, (N, N), label or "GQSP")


def evolve_gqsp(
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
    lam: float,
    stacked: BlockEncoding,
    state: NDArray[np.complexfloating],
    *,
    selector: bool = False,
) -> NDArray[np.complex128]:
    r"""Return :func:`gqsp_circuit`'s circuit applied to ``state``, one controlled walk at a time.

    Each step is one :func:`~pihm.circuits.primitives.evolve` of the controlled walk (compiled
    once) followed by that step's rotation, so no circuit ever holds the :math:`d` repetitions.
    Every :data:`PROGRESS_SECONDS` it logs the step reached, the mean step and the time left.

    Parameters
    ----------
    theta, phi, lam
        A phase triple.
    stacked
        :math:`U_R` (:func:`pihm.circuits.compose.stack_rows`).
    state
        A statevector over the walk's own qubits (``stacked``'s) and the signal qubit above them,
        plus, when ``selector`` is true, the selector qubit above the signal.
    selector
        Apply the whole circuit only where the selector qubit is :math:`|1\rangle` (the readout's
        LCU combiner): the rotations are controlled on the selector, the walk's
        reflections on the selector and the signal, and :math:`\lambda` is a phase gate on the
        selector.

    Returns
    -------
    numpy.ndarray
        The evolved statevector, the same length as ``state``.
    """
    degree = _degree(theta, phi)
    n_walk = stacked.n_qubits
    signal = n_walk
    width = n_walk + (2 if selector else 1)
    if state.shape != (1 << width,):
        raise ValueError(f"expected a length-{1 << width} state, got shape {state.shape}")

    rotation: Callable[[int], QuantumCircuit]
    if selector:
        select = n_walk + 1
        walk_step = compile_circuit(controlled_walk(stacked, n_controls=2, ctrl_state=0b10))

        def rotation(k: int) -> QuantumCircuit:
            qc = QuantumCircuit(width)
            R = controlled_matrix(rotation_gate(theta[k], phi[k]), 1, 1)
            qc.append(unitary_gate(R, label=f"cR{k}"), [select, signal])
            return qc

        first = rotation(0)
        first.p(float(lam), select)
    else:
        walk_step = compile_circuit(controlled_walk(stacked))

        def rotation(k: int) -> QuantumCircuit:
            qc = QuantumCircuit(width)
            qc.append(unitary_gate(rotation_gate(theta[k], phi[k]), label=f"R{k}"), [signal])
            return qc

        first = rotation(0)
        first.global_phase = float(lam)

    evolved = evolve(first, state)
    started = reported = time.perf_counter()
    for k in range(1, degree + 1):
        step = walk_step.copy()
        step.compose(rotation(k), inplace=True)
        evolved = evolve(step, evolved)
        now = time.perf_counter()
        if now - reported >= PROGRESS_SECONDS:
            reported = now
            per_step = (now - started) / k
            logger.info(
                "GQSP step %d of %d: %.3g s a step, %.3g h left",
                k,
                degree,
                per_step,
                per_step * (degree - k) / 3600.0,
            )
    return evolved


def apply_gqsp(
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
    lam: float,
    stacked: BlockEncoding,
    vector: NDArray[np.complexfloating],
    *,
    rescale: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Return :math:`\alpha\,\Pi_{\rm out}U\Pi_{\rm in}|{\rm vector}\rangle` for the GQSP circuit.

    What :func:`~pihm.circuits.primitives.apply` reads from :func:`gqsp_circuit`'s block
    encoding, computed by :func:`evolve_gqsp` without assembling the :math:`d`-fold circuit:
    ``vector`` on the system register, every other qubit :math:`|0\rangle`, and the leading
    amplitudes read back, scaled by :math:`\alpha=\text{rescale}`.

    Parameters
    ----------
    theta, phi, lam, stacked, rescale
        As :func:`gqsp_circuit`.
    vector
        Length ``stacked.shape[1]``.

    Returns
    -------
    numpy.ndarray
        Length ``stacked.shape[1]``: the target polynomial in :math:`H'` applied to ``vector``.
    """
    N = 1 << _system_qubits(stacked)
    if vector.shape != (N,):
        raise ValueError(f"expected a length-{N} vector, got shape {vector.shape}")
    full = np.zeros(1 << (stacked.n_qubits + 1), dtype=np.complex128)
    full[:N] = vector
    evolved = evolve_gqsp(theta, phi, lam, stacked, full)
    result: NDArray[np.complex128] = rescale * evolved[:N]
    return result
