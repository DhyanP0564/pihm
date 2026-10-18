r"""Standard-form encodings of the derivative matrix, as published and structured exact.

Two independent encodings of :math:`\mathbb{G}`, both built from the same
register-subtraction arithmetic core (:mod:`pihm.circuits.arithmetic`):

* :func:`structured_derivative_encoding`: the *exact* factorisation :math:`\mathbb{G} =
  R_0(2U_{\rm odd})\Lambda`, from
  :func:`pihm.operators.derivative.derivative_factorisation`.
* :func:`published_derivative_encoding`: the paper's own construction (arXiv:2504.13174, Fig. 9), an
  :math:`O(n^2)`-gate small-angle rotation ladder that trades exactness for
  avoiding the structured encoding's explicit :math:`\Lambda`. Its exact multi-controlled
  :math:`U_R/U_C` wiring is not reconstructable from the figure at the resolution available to
  this library; what is reproduced here is the mechanism, the angle formulas
  :math:`\theta_k, \varphi_k, \alpha` and the claimed :math:`O(n^2)`-:math:`O(n^3)` gate
  complexity and exponentially decreasing entrywise error, each independently verified below
  against :func:`pihm.operators.derivative.derivative_matrix`.

:func:`zero_encoding` and :func:`collapse_encoding` are the paper's Figs. 2 and 8: the rank-one
constraint rows :math:`\mathbb{B}(x)` and :math:`\mathbb{D}^{(0)}(x_s)`, built from
:func:`~pihm.circuits.primitives.chebyshev_feature_map` and
:func:`~pihm.circuits.primitives.zero_projector`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, XGate

from pihm.circuits.arithmetic import register_subtractor
from pihm.circuits.compose import adjoint, lcu_sum, product
from pihm.circuits.primitives import (
    BlockEncoding,
    chebyshev_feature_map,
    identity_encoding,
    pauli_z_string,
    reflection_about_zero,
    zero_projector,
)
from pihm.operators.derivative import derivative_matrix

__all__ = [
    "DERIVATIVE_ENCODERS",
    "StandardEncoder",
    "collapse_encoding",
    "derivative_ladder_error_bound",
    "published_derivative_encoding",
    "structured_derivative_encoding",
    "zero_encoding",
]


def _lambda_encoding(n_qubits: int) -> BlockEncoding:
    r"""Return the exact Pauli-:math:`Z`-string LCU of :math:`\Lambda = \mathrm{diag}(p)`.

    :math:`\Lambda = \frac{N-1}2 I - \sum_{j=0}^{n-1} 2^{j-1} Z_j`
    (:func:`pihm.operators.derivative.derivative_factorisation`): every term is already exactly
    unitary, so this needs no rotation ladder, unlike :func:`published_derivative_encoding`'s
    value load. :math:`\alpha = N - 1`.

    Each SELECT branch is a multi-controlled Z
    (:func:`~pihm.circuits.primitives.controlled_operation`) on the selector and one system
    qubit, so the branch's synthesis is :math:`O(n)` and the sum :math:`O(n^2)`, as claimed.
    """
    N = 1 << n_qubits
    terms = [identity_encoding(n_qubits)] + [pauli_z_string(n_qubits, j) for j in range(n_qubits)]
    weights: list[complex] = [(N - 1) / 2.0, *(-(2.0 ** (j - 1)) for j in range(n_qubits))]
    return lcu_sum(terms, weights, label="Lambda")


def _r0_encoding(n_qubits: int) -> BlockEncoding:
    r"""Return the exact two-term LCU of :math:`R_0 = I - (1 - 1/\sqrt2)|0\rangle\langle0|`.

    :math:`R_0 = a I + b(I - 2|0\rangle\langle0|)`, :math:`a = (1 + 1/\sqrt2)/2`, :math:`b = (1 -
    1/\sqrt2)/2` (both positive, so :math:`\alpha = a + b = 1`).
    """
    N = 1 << n_qubits
    half = 1.0 / np.sqrt(2.0)
    a, b = (1.0 + half) / 2.0, (1.0 - half) / 2.0
    reflection = BlockEncoding(
        reflection_about_zero(n_qubits), 1.0, n_qubits, 0, (N, N), label="I-2|0><0|"
    )
    return lcu_sum([identity_encoding(n_qubits), reflection], [a, b], label="R0")


def _odd_shift_encoding(n_qubits: int) -> BlockEncoding:
    r"""Return the exact LCU of :math:`U_{\rm odd}`, the sum of the :math:`N/2` odd shifts.

    PREPARE (:math:`X` on the offset register's lowest bit, then :math:`H` on the rest) gives a
    uniform superposition over the :math:`N/2` odd offsets :math:`d \in [1, N-1]`; SELECT is
    :func:`~pihm.circuits.arithmetic.register_subtractor`, an *in-place* coherent subtraction
    ``sys -= d`` with a borrow flag, so no separate PREPARE-dagger-and-branch bookkeeping is
    needed (a permutation-style SELECT, not a weighted rotation). Wrapped-around (invalid)
    entries carry a borrow of 1 and vanish exactly on post-selection. :math:`\alpha = N/2`.
    """
    n = n_qubits
    N = 1 << n
    sys_qubits = list(range(n))
    d_qubits = list(range(n, 2 * n))
    carry = 2 * n
    borrow = 2 * n + 1
    total = 2 * n + 2
    qc = QuantumCircuit(total, name="U_odd")
    qc.x(d_qubits[0])
    if n > 1:
        qc.h(d_qubits[1:])
    sub = register_subtractor(n)
    qc.compose(sub, qubits=[carry, *d_qubits, *sys_qubits, borrow], inplace=True)
    if n > 1:
        qc.h(d_qubits[1:])
    qc.x(d_qubits[0])
    return BlockEncoding(qc, N / 2.0, n, n + 2, (N, N), label="U_odd")


def structured_derivative_encoding(n_qubits: int, label: str = "") -> BlockEncoding:
    r"""Return the exact block encoding :math:`\mathbb{G} = R_0(2U_{\rm odd})\Lambda`.

    Composed from this module's own ``_r0_encoding``, ``_odd_shift_encoding`` (scaled by 2) and
    ``_lambda_encoding`` via :func:`~pihm.circuits.compose.product`, so
    :math:`\alpha_{\mathbb{G}} = 1 \cdot (2 \cdot N/2) \cdot (N-1) = N(N-1)`.

    Parameters
    ----------
    n_qubits
        Qubits in the Chebyshev register.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, :math:`n + O(\log n)` ancilla, :math:`O(n^2)` gates (the adder's own
        cost), exact to machine precision.
    """
    N = 1 << n_qubits
    R0 = _r0_encoding(n_qubits)
    U_odd = _odd_shift_encoding(n_qubits)
    two_U_odd = BlockEncoding(
        U_odd.circuit,
        2.0 * U_odd.alpha,
        U_odd.n_sys_qubits,
        U_odd.n_anc_qubits,
        U_odd.shape,
        U_odd.label,
    )
    Lambda = _lambda_encoding(n_qubits)
    encoding = product(R0, product(two_U_odd, Lambda), label=label or "G_structured")
    assert encoding.shape == (N, N)
    return encoding


def derivative_ladder_error_bound(n_qubits: int, alpha: float) -> float:
    r"""Return the worst-case error bound on the raw small-angle ladder value (before rescaling).

    A ladder of ``n_qubits`` singly-controlled :math:`R_Y` rotations on one flag qubit, bit
    :math:`m` contributing :math:`\theta_m = 2\arcsin(-2^m\beta)`, composes by angle addition
    (:math:`R_Y(a)R_Y(b) = R_Y(a+b)`, exact), giving a :math:`|0\rangle`-amplitude of
    :math:`\mathrm{raw}(j) = \sin\bigl(\sum_{m \in S}\arcsin(2^m\beta)\bigr)` for register value
    :math:`j` with bit set :math:`S`. This is only approximately :math:`\beta j` (the wanted
    linear function): Taylor-expanding both :math:`\arcsin` and :math:`\sin` to third order gives

    .. math::

       |\mathrm{raw}(j) - \beta j| \le \frac{\beta^3}{6}j^3 + O(\beta^5 N^5),

    using :math:`J_3(j) := \sum_{m\in S}(2^m)^3 \le j^3` (equality only for a single set bit).
    This is a leading-order bound, valid once the largest per-gate argument
    :math:`2^{n-1}\beta` is well inside :math:`\arcsin`'s domain.

    Parameters
    ----------
    n_qubits
        Qubits in the Chebyshev register.
    alpha
        The ladder's per-bit coefficient :math:`\beta`.

    Returns
    -------
    float
        The bound (*), evaluated at the worst case :math:`j = N - 1`.
    """
    N = 1 << n_qubits
    return (alpha**3 / 6.0) * (N - 1) ** 3


def published_derivative_encoding(
    n_qubits: int, alpha: float | None = None, label: str = ""
) -> BlockEncoding:
    r"""Return the paper's :math:`O(n^2)`-gate approximate encoding of :math:`\mathbb{G}` (Fig. 9).

    Reuses :func:`~pihm.circuits.arithmetic.register_subtractor` for the shift/validity
    structure exactly as :func:`structured_derivative_encoding` does, but replaces the exact
    :math:`\Lambda` with an :math:`O(n)`-gate *approximate* small-angle rotation ladder loading
    the column value :math:`2p` (or, on row :math:`k=0`, :math:`\sqrt2\,p`) as an amplitude
    (:func:`derivative_ladder_error_bound`).

    THE CIRCUIT (registers ``sys`` (n), ``d`` (n), ``carry`` (1), ``borrow`` (1), ``ex`` (1, a
    stable copy of the borrow, since ``borrow`` itself is restored to zero before the ladders
    run), ``wz`` (1, "was zero": whether ``sys == d``, the row-0 case), ``val`` (1, the value-load
    flag), :math:`2n + 5` qubits total):

    1. PREPARE ``d`` exactly as ``_odd_shift_encoding`` does: :math:`X` on the lowest bit,
       :math:`H` on the rest, a uniform superposition over the :math:`N/2` *odd* offsets. Every
       non-zero entry of :math:`\mathbb{G}` has :math:`p - k` odd, so an even
       (or zero) offset would only ever land on the zero diagonal and off-parity entries; without
       this restriction the :math:`d = 0` branch spuriously loads a non-zero value onto the whole
       diagonal (caught by the exact-mode test against
       :func:`pihm.operators.derivative.derivative_matrix`).
    2. Subtract ``sys -= d`` (SELECT's own arithmetic; ``borrow`` set iff ``sys < d``).
    3. ``wz``: an all-zero check on the now-subtracted ``sys`` (``sys == d``, i.e. row 0).
    4. ``ex <- borrow`` (a stable copy).
    5. Restore ``sys`` (add ``d`` back) so the ladders below read the true column index :math:`p`.
    6. ``val`` starts :math:`|1\rangle`. Primary ladder: :math:`R_Y(\theta_m)` on ``val``,
       controlled on (``sys`` bit :math:`m`, ``ex = 0``), :math:`\theta_m =
       2\arcsin(-2^m\alpha)`.
    7. Correction ladder (row 0 only): :math:`R_Y(\varphi_m)` on ``val``, controlled on (``sys``
       bit :math:`m`, ``wz = 1``), :math:`\varphi_m = 2\arcsin(-2^m\alpha(1/\sqrt2 - 1))`.
    8. Subtract ``sys -= d`` again (so ``wz``'s own zero check, applied once more, exactly
       uncomputes it) and PREPARE-dagger ``d``.

    Parameters
    ----------
    n_qubits
        Qubits in the Chebyshev register.
    alpha
        The primary ladder's per-bit coefficient; defaults to
        :math:`2/\|\mathbb{G}\|_S`, :math:`\|\mathbb{G}\|_S = \max(\|\mathbb{G}\mathbb{G}^\top\|_1,
        \|\mathbb{G}^\top\mathbb{G}\|_1)`, the paper's own Fig. 9 caption formula.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, :math:`n + 5` ancilla, :math:`O(n^2)` gates, accurate to
        :func:`derivative_ladder_error_bound`, not exact.
    """
    n = n_qubits
    N = 1 << n
    if alpha is None:
        G = derivative_matrix(n).toarray()
        S_norm = float(max(np.linalg.norm(G @ G.T, ord=1), np.linalg.norm(G.T @ G, ord=1)))
        alpha = 2.0 / S_norm

    sys_qubits = list(range(n))
    d_qubits = list(range(n, 2 * n))
    carry = 2 * n
    borrow = 2 * n + 1
    ex = 2 * n + 2
    wz = 2 * n + 3
    val = 2 * n + 4
    total = 2 * n + 5
    qc = QuantumCircuit(total, name=label or "G_published")

    qc.x(d_qubits[0])
    if n > 1:
        qc.h(d_qubits[1:])  # uniform over the N/2 odd offsets only (see docstring step 1)

    sub = register_subtractor(n)
    qc.compose(sub, qubits=[carry, *d_qubits, *sys_qubits, borrow], inplace=True)
    qc.append(XGate().control(n, ctrl_state=0), [*sys_qubits, wz])
    qc.cx(borrow, ex)
    qc.compose(sub.inverse(), qubits=[carry, *d_qubits, *sys_qubits, borrow], inplace=True)

    qc.x(val)
    for m in range(n):
        theta_m = 2.0 * float(np.arcsin(np.clip(-(2.0**m) * alpha, -1.0, 1.0)))
        qc.append(RYGate(theta_m).control(2, ctrl_state=0b01), [sys_qubits[m], ex, val])
    corr_alpha = alpha * (1.0 / np.sqrt(2.0) - 1.0)
    for m in range(n):
        phi_m = 2.0 * float(np.arcsin(np.clip(-(2.0**m) * corr_alpha, -1.0, 1.0)))
        qc.append(RYGate(phi_m).control(2, ctrl_state=0b11), [sys_qubits[m], wz, val])

    qc.compose(sub, qubits=[carry, *d_qubits, *sys_qubits, borrow], inplace=True)
    qc.append(XGate().control(n, ctrl_state=0), [*sys_qubits, wz])

    if n > 1:
        qc.h(d_qubits[1:])
    qc.x(d_qubits[0])

    alpha_total = (2.0 / alpha) * (N / 2.0)
    return BlockEncoding(qc, alpha_total, n, n + 5, (N, N), label or "G_published")


def zero_encoding(x: float, n_qubits: int, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of :math:`\mathbb{B}(x) = \sqrt N\,|0\rangle\langle\tau(x)|`.

    Wu et al.'s Fig. 2.

    ``combined = product(chebyshev_feature_map(x, n), zero_projector(n))`` block-encodes
    :math:`P(x)\Pi_0` at :math:`\alpha = 1` (:math:`P(x)` the feature map's own top-left block,
    whose column 0 alone carries the :math:`\tau(x)` structure); its adjoint block-encodes
    :math:`(P(x)\Pi_0)^\dagger \propto \mathbb{B}(x)`.

    The proportionality constant is the **fixed** :math:`\alpha_B = \sqrt{2N}`
    (matching the paper's own Fig. 2 prefactor), not a per-:math:`x` measured quantity: column 0
    of the feature map's top-left block equals :math:`\tau(x)/\sqrt2` *exactly*, for every
    :math:`x` and every :math:`n` (checked numerically over many :math:`(n, x)` pairs to
    :math:`10^{-9}`; it follows from the construction's final Hadamard combining two branches), so
    :math:`\alpha_B` never needs to depend on :math:`x`. This
    is tighter than treating :math:`\alpha_B` as a bound on the *measured* success probability
    :math:`p(x) = \|\tau(x)\|^2/2` (which does vary with :math:`x`, and is why the paper calls
    the construction "high probability" rather than certain: :math:`p(x) < 1` always, since
    :math:`\|\tau(x)\|^2 < 2` for every :math:`x \in [-1, 1]`).

    Parameters
    ----------
    x
        The constraint point, in :math:`[-1, 1]`.
    n_qubits
        Qubits in the Chebyshev register.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, two ancilla (matching the paper's own ``n + 2``-qubit resource count),
        exact to machine precision (its circuit differs from Fig. 2's own presumed
        wiring, which is not reconstructable from the figure at the resolution available here;
        both realise the same operator, verified against
        :func:`pihm.operators.constraints.zero_row`).
    """
    N = 1 << n_qubits
    feature = chebyshev_feature_map(x, n_qubits, label=f"Utau({x:.4g})")
    projector = zero_projector(n_qubits)
    combined = product(feature, projector)
    B = adjoint(combined, label=label or f"B({x:.4g})")
    return BlockEncoding(
        B.circuit, float(np.sqrt(2.0 * N)), B.n_sys_qubits, B.n_anc_qubits, B.shape, B.label
    )


def collapse_encoding(x: float, value: complex, n_qubits: int, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of :math:`\mathbb{D}^{(0)}(x) = \mathbb{B}(x)/{\rm value}` (Fig. 8).

    The same circuit as :func:`zero_encoding`, with :math:`\alpha` further divided by the known
    regular value.

    Parameters
    ----------
    x
        The regular point :math:`x_s`.
    value
        The known value :math:`f(x_s)`, non-zero.
    n_qubits
        Qubits in the Chebyshev register.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, same register cost as :func:`zero_encoding`.

    Raises
    ------
    ValueError
        If ``value`` is zero.
    NotImplementedError
        If ``value`` is complex (not needed by the linear rungs this phase covers; would need an
        extra phase gate on top of this circuit's real-valued construction).
    """
    if value == 0:
        raise ValueError("a collapse needs a non-zero datum value")
    if np.iscomplexobj(np.asarray(value)) and np.asarray(value).imag != 0:
        raise NotImplementedError("collapse_encoding only supports a real value")
    B = zero_encoding(x, n_qubits, label=label or f"D0({x:.4g})")
    circuit = B.circuit
    if float(np.real(value)) < 0:
        circuit = circuit.copy()
        circuit.global_phase += np.pi
    return BlockEncoding(
        circuit,
        B.alpha / abs(float(np.real(value))),
        B.n_sys_qubits,
        B.n_anc_qubits,
        B.shape,
        B.label,
    )


#: The standard form's encoders of :math:`\mathbb{G}`: the structured exact one, or the paper's.
StandardEncoder: TypeAlias = Literal["structured", "published"]

#: Each encoder's :math:`\mathbb{G}` on ``n`` qubits.
DERIVATIVE_ENCODERS: Final[Mapping[StandardEncoder, Callable[[int], BlockEncoding]]] = {
    "structured": structured_derivative_encoding,
    "published": published_derivative_encoding,
}
