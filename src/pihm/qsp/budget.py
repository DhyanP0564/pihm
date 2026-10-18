r"""The state-preparation error budget, term by term.

A prepared state reaches the field through a chain of approximations, and each link is measured
as a distance between unit states, the global phase chosen to minimise it,
:math:`d(a,b)=\min_\phi\|a-e^{i\phi}b\|`, which is a metric on states:

1. **filter**: from the target (:func:`pihm.qsp.prepare.filter_target`) to the untruncated filter's
   output, :math:`e^{-\tau(1+H')}\varphi_0` normalised, computed exactly from the residual's SVD.
   It is what the imaginary time leaves of the excited states. The minimax filter is its own
   polynomial, :math:`p_d(H')\varphi_0`.
2. **truncation**: from there to T2's state, the filter's certified truncation applied. The
   minimax filter truncates nothing, and the term is the rounding of T2's Chebyshev recurrence:
   up to about :math:`5\times10^{-8}` at degrees past :math:`10^4`, where the spectral
   evaluation agrees with a 50-digit one to about :math:`10^{-12}`.
3. **phases**: from T2's state to the state the GQSP phases realise, their realised polynomial
   (:func:`pihm.qsp.phases.realised_coefficients`) applied in T2's place. T2 applies the target
   polynomial, so this is what the circuit adds.

The budget's remaining terms are exactly zero here: every encoding is exact, so
:math:`d\,\varepsilon_{\rm BE}=0`, and the field is read out classically, so
:math:`\varepsilon_{\rm sample}=0`. By the triangle inequality T2's own distance from the target is
at most the first two terms, and the circuit's at most all three; the exact readout bound
(:class:`pihm.classical.ReadoutBound`) turns a distance into a field error.

Each distance is formed as the norm of a difference, never as
:math:`\sqrt{2-2|\langle a|b\rangle|}`, which cancels to a floor of about :math:`10^{-8}`, the
size of the terms themselves.

The budget is taken on the physical columns (:attr:`pihm.assemble.Residual.physical_columns`):
the padding decouples exactly, and every state here is zero on it. Its spectral decomposition is
the dense reference's own where the reference is dense, and a dense SVD of the physical columns
past that ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm.classical import reference_state
from pihm.qsp.minimax import MinimaxFilter, minimax_values
from pihm.qsp.phases import realised_coefficients
from pihm.qsp.prepare import T2Result, apply_qite_filter, filter_target

__all__ = ["StateBudget", "state_budget", "state_distance"]


def state_distance(a: NDArray[np.complexfloating], b: NDArray[np.complexfloating]) -> float:
    r"""Return :math:`\min_\phi\|a-e^{i\phi}b\|` for unit vectors, or ``nan`` if either is zero."""
    if not np.any(a) or not np.any(b):
        return math.nan
    overlap = complex(np.vdot(b, a))
    phase = overlap / abs(overlap) if overlap != 0 else 1.0
    return float(np.linalg.norm(a - phase * b))


def _unit(vector: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    """Return ``vector`` normalised, or zero if it is zero."""
    norm = float(np.linalg.norm(vector))
    unit: NDArray[np.complex128] = np.asarray(
        vector / norm if norm > 0.0 else vector, np.complex128
    )
    return unit


@dataclass(frozen=True)
class StateBudget:
    r"""The state terms for one T2 result (the module docstring).

    Attributes
    ----------
    filter, truncation, phases
        The distances along the chain: target to the untruncated filter's output, to T2's state,
        to the state the phases realise. ``nan`` where a state underflows to zero.
    measured
        T2's own distance from the target, at most ``filter + truncation``.
    """

    filter: float
    truncation: float
    phases: float
    measured: float

    @property
    def t2(self) -> float:
        """The budget for T2's state, ``filter + truncation``."""
        return self.filter + self.truncation

    @property
    def circuit(self) -> float:
        """The budget for the circuit's state, all three terms."""
        return self.filter + self.truncation + self.phases


def state_budget(result: T2Result) -> StateBudget:
    """Return the state terms for a T2 result (the module docstring)."""
    residual = result.residual
    R = residual.matrix()
    columns = residual.physical_columns
    spectral = result.reference
    if spectral.right_vectors is None:
        spectral = reference_state(sp.csr_array(R[:, columns]))
    vh = spectral.right_vectors
    assert vh is not None  # a dense reference keeps its decomposition
    singular = spectral.singular_values[::-1]
    phi0 = np.asarray(result.initial, dtype=np.complex128)

    _gap, project = filter_target(residual, result.reference)
    target = project(phi0)[columns]
    qf = result.filter
    if isinstance(qf, MinimaxFilter):
        x = 2.0 * (singular / result.alpha_r) ** 2 - 1.0
        weights = minimax_values(x, qf.degree, qf.x_excited)
    else:
        weights = np.exp(-2.0 * qf.tau * (singular / result.alpha_r) ** 2)
    untruncated = _unit(vh.conj().T @ (weights * (vh @ phi0[columns])))

    realised = realised_coefficients(result.phases.theta, result.phases.phi, result.phases.lam)
    realised = realised * result.phases.rescale
    circuit = _unit(
        apply_qite_filter(R, result.alpha_r, realised.real, phi0)
        + 1j * apply_qite_filter(R, result.alpha_r, realised.imag, phi0)
    )[columns]
    prepared = result.state[columns]
    return StateBudget(
        filter=state_distance(untruncated, target),
        truncation=state_distance(prepared, untruncated),
        phases=state_distance(circuit, prepared),
        measured=state_distance(prepared, target),
    )
