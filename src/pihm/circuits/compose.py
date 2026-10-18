r"""Composition of block encodings: products, sums, the stacked residual and its reflection.

Every primitive here works at the level of :class:`~pihm.circuits.primitives.BlockEncoding`
circuits (never a materialised dense matrix), following the composition rules.

* :func:`product`, :func:`adjoint`: :math:`X \mapsto X^\dagger`, :math:`(X, Y) \mapsto XY`.
* :func:`lcu_sum`: :math:`\sum_i w_i X_i`, an ordinary linear combination of unitaries (used for
  :math:`\Lambda`'s Pauli-:math:`Z` terms and :math:`R_0`'s two-term identity/reflection sum in
  :mod:`pihm.circuits.standard`).
* :func:`pad_rows`: :math:`[X;\,0]`, a block moved onto a lifted axis's register (one more,
  idle, system qubit), where the paper's lifted rows live.
* :func:`stack_rows`: the *stacked residual* :math:`U_R`, block-encoding the
  tall :math:`R = [R_1; \dots; R_L]` with a row-select register that survives the sandwich (it is
  what the caller reads out), unlike :func:`lcu_sum`'s selector, which is always un-prepared away.
* :func:`stacked_reflection`, :func:`walk`: :math:`V = U_R^\dagger(2\Pi_{\rm mid} -
  I)U_R` and :math:`W = (2\Pi_{\rm in} - I)V`, where :math:`V`'s block is
  :math:`2H/\alpha_R^2 - I` exactly.
* :func:`controlled_walk`: :math:`W` controlled on further qubits, with only its two
  reflections carrying the control (:math:`CV = U_R^\dagger\,C(2\Pi - I)\,U_R`), as GQSP
  (:mod:`pihm.circuits.gqsp`) applies it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import GlobalPhaseGate, RYGate, XGate

from pihm.circuits.primitives import (
    BlockEncoding,
    controlled_operation,
    inverse_circuit,
    reflection_about_zero,
)

__all__ = [
    "adjoint",
    "controlled_walk",
    "embed_system",
    "lcu_sum",
    "pad_rows",
    "product",
    "real_state_preparation",
    "stack_rows",
    "stacked_reflection",
    "walk",
]


def _reflection_2pi_minus_i(n_qubits: int, label: str = "") -> QuantumCircuit:
    r"""Return :math:`2|0\rangle\langle0| - I`, the negative of :func:`reflection_about_zero`.

    :func:`~pihm.circuits.primitives.reflection_about_zero` builds :math:`I -
    2|0\rangle\langle0|` (what :math:`R_0`'s own LCU term needs); the projector convention
    :math:`(2\Pi_{a=0} - I)` that :func:`stacked_reflection` and :func:`walk` use is its
    negative, added here as a global phase of :math:`\pi` rather than a second gate sequence.
    """
    qc = reflection_about_zero(n_qubits, label=label)
    qc.global_phase += np.pi
    return qc


def adjoint(encoding: BlockEncoding, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of :math:`X^\dagger` from one of :math:`X`.

    Since :math:`\Pi_{\rm out} U \Pi_{\rm in}` uses Hermitian projectors,
    :math:`(\Pi_{\rm out} U \Pi_{\rm in})^\dagger = \Pi_{\rm in} U^\dagger \Pi_{\rm out}`, so
    ``encoding.circuit``'s exact inverse block-encodes :math:`X^\dagger` at the same
    :math:`\alpha`, with the input/output roles (hence ``shape``) swapped.

    Parameters
    ----------
    encoding
        The block encoding of :math:`X`.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(shape[1], shape[0])``, same :math:`\alpha` and register split.
    """
    rows, cols = encoding.shape
    return BlockEncoding(
        inverse_circuit(encoding.circuit),
        encoding.alpha,
        encoding.n_sys_qubits,
        encoding.n_anc_qubits,
        (cols, rows),
        label or f"({encoding.label})†",
    )


def product(left: BlockEncoding, right: BlockEncoding, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of :math:`Z = XY` from block encodings of :math:`X` and :math:`Y`.

    ``right`` runs first, then ``left``, both on the system register and **one shared ancilla
    register**.
    Between them, a flag qubit is set wherever ``right``'s ancilla is not :math:`|0\rangle`: those
    branches have already failed, so ``left`` may scramble their ancilla freely, and they stay
    excluded because the flag is part of the post-selected ancilla. Where ``right`` succeeded,
    ``left`` finds its ancilla in :math:`|0\rangle`, as its block encoding needs. So the product is
    exact at :math:`\alpha = \alpha_{\rm left}\alpha_{\rm right}` with
    :math:`n_{\rm anc} = \max(n_{\rm anc,left}, n_{\rm anc,right}) + 1` (no flag when ``right``
    has no ancilla, since it cannot fail).

    Parameters
    ----------
    left, right
        Block encodings sharing ``n_sys_qubits``, with ``left.shape[1] == right.shape[0]``.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(left.shape[0], right.shape[1])``.

    Raises
    ------
    ValueError
        If the system registers or the inner shape do not match.
    """
    if left.n_sys_qubits != right.n_sys_qubits:
        raise ValueError("product() requires both encodings to share the system register size")
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"shape mismatch for X@Y: {left.shape} @ {right.shape}")
    n_sys = left.n_sys_qubits
    shared = max(left.n_anc_qubits, right.n_anc_qubits)
    flagged = right.n_anc_qubits > 0
    total = n_sys + shared + int(flagged)
    name = label or f"({left.label})·({right.label})"
    qc = QuantumCircuit(total, name=name)
    qc.compose(right.circuit, qubits=range(n_sys + right.n_anc_qubits), inplace=True)
    if flagged:
        flag = total - 1
        ancilla = list(range(n_sys, n_sys + right.n_anc_qubits))
        qc.x(flag)
        qc.append(XGate().control(len(ancilla), ctrl_state=0), [*ancilla, flag])
    qc.compose(left.circuit, qubits=range(n_sys + left.n_anc_qubits), inplace=True)
    return BlockEncoding(
        qc, left.alpha * right.alpha, n_sys, total - n_sys, (left.shape[0], right.shape[1]), name
    )


def _next_pow2(m: int) -> int:
    """Return the smallest power of two at least ``m`` (1 for ``m <= 1``)."""
    return 1 if m <= 1 else 1 << (m - 1).bit_length()


def _rotation_tree(
    weights: NDArray[np.float64], signs: NDArray[np.float64] | None
) -> QuantumCircuit:
    r"""Return the :math:`R_Y` tree preparing :math:`\sum_is_i\sqrt{w_i/\alpha}|i\rangle` from zero.

    From the highest qubit down, each rotation splits a subtree's weight between its two halves,
    controlled on the qubits above it selecting that subtree; the lowest level's rotations also
    carry the signs :math:`s_i` (all :math:`+1` when ``signs`` is ``None``). ``weights`` has a
    power-of-two length.
    """
    s_qubits = weights.size.bit_length() - 1
    qc = QuantumCircuit(s_qubits)
    for q in reversed(range(s_qubits)):
        above = s_qubits - 1 - q
        for prefix in range(1 << above):
            block = weights[prefix << (q + 1) : (prefix + 1) << (q + 1)]
            low, high = block[: 1 << q].sum(), block[1 << q :].sum()
            low_root, high_root = np.sqrt(low), np.sqrt(high)
            if q == 0 and signs is not None:
                low_root *= signs[prefix << 1]
                high_root *= signs[(prefix << 1) + 1]
            if high == 0.0 and low_root >= 0.0:
                continue
            angle = 2.0 * np.arctan2(high_root, low_root)
            if above == 0:
                qc.ry(angle, q)
            else:
                gate = RYGate(angle).control(above, ctrl_state=prefix)
                qc.append(gate, [*range(q + 1, s_qubits), q])
    return qc


def _prepare_circuit(weights: Sequence[float]) -> QuantumCircuit:
    r"""Return PREPARE, :math:`|0\rangle\mapsto\sum_i\sqrt{w_i/\alpha}\,|i\rangle`, gate by gate.

    :math:`w_i` the weights (non-negative), :math:`\alpha = \sum_i w_i`, padded to the next
    power-of-two dimension. The amplitudes are real and non-negative, so a binary tree of
    :math:`R_Y` rotations prepares them (``_rotation_tree``). Only the action on
    :math:`|0\rangle` matters (every other input feeds a branch the ancilla-zero post-selection
    discards).

    Parameters
    ----------
    weights
        Non-negative weights, not all zero.

    Returns
    -------
    qiskit.QuantumCircuit
        On :math:`\lceil\log_2 L\rceil` qubits, :math:`L` = ``len(weights)``.
    """
    values = np.asarray(weights, dtype=float)
    if np.any(values < 0):
        raise ValueError("PREPARE needs non-negative weights")
    if values.sum() <= 0:
        raise ValueError("PREPARE needs a strictly positive total weight")
    padded = np.zeros(_next_pow2(values.size))
    padded[: values.size] = values
    qc = _rotation_tree(padded, None)
    qc.name = "PREPARE"
    return qc


def real_state_preparation(amplitudes: ArrayLike, label: str = "") -> QuantumCircuit:
    r"""Return a circuit taking :math:`|0\rangle` to the unit vector along ``amplitudes``.

    A real vector, any signs, of power-of-two length: ``_rotation_tree`` on the squared
    amplitudes, the signs carried by the lowest level's rotations. :math:`2^m-1` rotations at
    most, each controlled on the qubits above its own, one fewer for every subtree that is zero.

    Parameters
    ----------
    amplitudes
        Real, not all zero; normalised here.
    label
        Circuit name.

    Returns
    -------
    qiskit.QuantumCircuit
        On :math:`m = \log_2` ``len(amplitudes)`` qubits.
    """
    values = np.asarray(amplitudes, dtype=float)
    size = values.size
    if size == 0 or size & (size - 1):
        raise ValueError("real_state_preparation needs a power-of-two number of amplitudes")
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("real_state_preparation needs a nonzero vector")
    unit = values / norm
    qc = _rotation_tree(unit**2, np.where(unit < 0.0, -1.0, 1.0))
    qc.name = label or "S"
    return qc


def _select(
    qc: QuantumCircuit,
    branch: QuantumCircuit,
    selector: Sequence[int],
    index: int,
    qubits: Sequence[int],
    phase: complex = 1.0,
) -> None:
    r"""Append ``branch`` times ``phase``, controlled on ``selector`` equalling ``index``.

    Each instruction is controlled as it stands
    (:func:`~pihm.circuits.primitives.controlled_operation`), so a multi-controlled gate stays one,
    and the branch's phase (with its own global phase, which control makes relative) is a
    controlled global-phase gate on the selector alone. ``qubits[k]`` is where ``branch``'s qubit
    :math:`k` lands.
    """
    angle = float(np.angle(phase)) + float(branch.global_phase)
    if not selector:
        qc.compose(branch, qubits=list(qubits), inplace=True)
        qc.global_phase += float(np.angle(phase))
        return
    for inst in branch.data:
        local = [qubits[branch.find_bit(q).index] for q in inst.qubits]
        qc.append(controlled_operation(inst.operation, len(selector), index), [*selector, *local])
    if np.mod(angle, 2.0 * np.pi) != 0.0:
        qc.append(GlobalPhaseGate(angle).control(len(selector), ctrl_state=index), list(selector))


def lcu_sum(
    terms: Sequence[BlockEncoding], weights: Sequence[complex], label: str = ""
) -> BlockEncoding:
    r"""Return a block encoding of :math:`\sum_i w_i X_i` from block encodings of each :math:`X_i`.

    A standard linear-combination-of-unitaries construction: PREPARE (amplitudes
    :math:`\sqrt{|w_i|\alpha_i/\alpha}`, :math:`\alpha = \sum_i|w_i|\alpha_i`) on a fresh
    selector register, SELECT (each term's own circuit, controlled on the selector equalling its
    index, built instruction by instruction so no dense matrix larger than one instruction plus
    the selector is ever materialised), then PREPARE-dagger. Every term shares one ancilla
    register sized to the largest term's own ancilla (unused high qubits of it are simply never
    touched by a smaller term's branch).

    Parameters
    ----------
    terms
        Block encodings sharing ``n_sys_qubits`` and ``shape``.
    weights
        One (possibly negative or complex) weight per term.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``terms[0].shape``, :math:`\alpha = \sum_i |w_i|\alpha_i`.

    Raises
    ------
    ValueError
        If ``terms``/``weights`` disagree in length, share no positive-weight term, or the terms
        do not share a system register size and shape.
    """
    if len(terms) != len(weights):
        raise ValueError("terms and weights must have the same length")
    nonzero = [(t, w) for t, w in zip(terms, weights, strict=True) if w != 0]
    if not nonzero:
        raise ValueError("lcu_sum requires at least one non-zero-weight term")
    n_sys = nonzero[0][0].n_sys_qubits
    shape = nonzero[0][0].shape
    for t, _ in nonzero:
        if t.n_sys_qubits != n_sys or t.shape != shape:
            raise ValueError("lcu_sum requires every term to share n_sys_qubits and shape")

    magnitudes = [abs(w) * t.alpha for t, w in nonzero]
    alpha = float(np.sum(magnitudes))
    n_anc_shared = max(t.n_anc_qubits for t, _ in nonzero)
    PREPARE = _prepare_circuit(magnitudes)
    s_qubits = PREPARE.num_qubits
    n_anc = s_qubits + n_anc_shared
    total = n_sys + n_anc
    name = label or "lcu_sum"
    selector = list(range(n_sys + n_anc_shared, total))

    qc = QuantumCircuit(total, name=name)
    qc.compose(PREPARE, qubits=selector, inplace=True)
    for i, (term, w) in enumerate(nonzero):
        _select(qc, term.circuit, selector, i, range(term.n_qubits), w / abs(w))
    qc.compose(inverse_circuit(PREPARE), qubits=selector, inplace=True)
    return BlockEncoding(qc, alpha, n_sys, n_anc, shape, name)


def pad_rows(encoding: BlockEncoding, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of :math:`[X;\,0]`: ``encoding``'s rows below one idle qubit.

    One more system qubit, the highest, that no gate touches: reading the output with it in
    :math:`|0\rangle` gives :math:`X`, and with it in :math:`|1\rangle` gives zero. This puts a
    block on a lifted axis's register, where the paper's lifted operator lives,
    at the same :math:`\alpha`; the embedding :math:`\mathbb{M}_1=\sqrt2\,[I;\,0]` is the padded
    identity at :math:`\alpha=\sqrt2`.

    Parameters
    ----------
    encoding
        A block encoding whose rows fill its system register.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(2 * shape[0], shape[1])``, one more system qubit, the same ancilla.

    Raises
    ------
    ValueError
        If ``encoding``'s rows do not fill its system register.
    """
    rows, cols = encoding.shape
    n_sys = encoding.n_sys_qubits
    if rows != 1 << n_sys:
        raise ValueError("pad_rows requires an encoding whose rows fill its system register")
    total = encoding.n_qubits + 1
    name = label or f"[{encoding.label}; 0]"
    qc = QuantumCircuit(total, name=name)
    qubits = [*range(n_sys), *range(n_sys + 1, total)]
    qc.compose(encoding.circuit, qubits=qubits, inplace=True)
    return BlockEncoding(
        qc, encoding.alpha, n_sys + 1, encoding.n_anc_qubits, (2 * rows, cols), name
    )


def embed_system(
    encoding: BlockEncoding, positions: Sequence[int], n_total: int, label: str = ""
) -> BlockEncoding:
    r"""Return ``encoding`` on the system qubits ``positions`` of an ``n_total``-qubit register.

    The identity acts on the other system qubits, so the result block-encodes the Kronecker
    product of ``encoding``'s matrix with the identity, in the qubit order ``positions`` gives
    (``positions[k]`` receives ``encoding``'s system qubit :math:`k`); its ancilla follow the
    system register. A product of such embeddings on disjoint positions is a Kronecker product
    of block encodings (:func:`product` shares their ancilla).

    Parameters
    ----------
    encoding
        A square block encoding whose rows fill its system register.
    positions
        Distinct qubits of the larger register, one per system qubit of ``encoding``.
    n_total
        System qubits of the larger register.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(2**n_total, 2**n_total)``, the same :math:`\alpha` and ancilla.

    Raises
    ------
    ValueError
        If ``encoding`` is not square on its register, or ``positions`` do not fit it.
    """
    n_sys = encoding.n_sys_qubits
    size = 1 << n_sys
    if encoding.shape != (size, size):
        raise ValueError("embed_system requires a square encoding on its whole register")
    if len(positions) != n_sys or len(set(positions)) != n_sys:
        raise ValueError("embed_system needs one distinct position per system qubit")
    if min(positions, default=0) < 0 or max(positions, default=-1) >= n_total:
        raise ValueError("embed_system positions must lie in the larger register")
    total = n_total + encoding.n_anc_qubits
    name = label or encoding.label
    qc = QuantumCircuit(total, name=name)
    qubits = [*positions, *range(n_total, total)]
    qc.compose(encoding.circuit, qubits=qubits, inplace=True)
    dim = 1 << n_total
    return BlockEncoding(qc, encoding.alpha, n_total, encoding.n_anc_qubits, (dim, dim), name)


def stack_rows(blocks: Sequence[BlockEncoding], label: str = "") -> BlockEncoding:
    r"""Return :math:`U_R`, block-encoding the tall stack :math:`R = [R_1; \dots; R_L]`.

    Unlike :func:`lcu_sum` (which un-prepares its selector, coherently *summing* the terms),
    this keeps the selector as part of the meaningful output: with a fresh row-select register
    of :math:`\lceil\log_2 L\rceil` qubits placed *between* the system register and the terms'
    own shared ancilla,

    .. math::

       \langle j|\langle 0_a|\,U_R\,|0\rangle|0_a\rangle = R_j/\alpha_R,
       \qquad \alpha_R = \Bigl(\sum_j\alpha_j^2\Bigr)^{1/2}

    PREPARE weights are :math:`\alpha_j^2` (a quadrature, :math:`\ell_2`
    combination, not :math:`\ell_1`), and there is no PREPARE-dagger. Reading the resulting
    :class:`~pihm.circuits.primitives.BlockEncoding` with ``n_sys_qubits = n_sys + s_qubits``
    (system and selector together) and ``n_anc_qubits`` = the terms' own shared ancilla treats
    the selector as *output*, exactly matching :func:`pihm.circuits.primitives.apply`'s
    ancilla-zero convention for the combined (row, block) index.

    Parameters
    ----------
    blocks
        The row blocks :math:`R_1, \dots, R_L`, block encodings sharing ``n_sys_qubits`` and
        ``shape``, whose rows fill the system register (``shape[0]`` is
        :math:`2^{\text{n\_sys\_qubits}}`): square, or lifted by :func:`pad_rows` or
        :func:`pihm.circuits.multiplication.multiplication_encoding`.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(M * rows, columns)``, :math:`M = 2^{\lceil\log_2 L\rceil}`, for the blocks' shared
        ``(rows, columns)``; ``n_sys_qubits = n_sys + s_qubits``, ``n_anc_qubits`` = the terms'
        shared ancilla width.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, their rows do not fill the system register, or they disagree in
        ``n_sys_qubits``/``shape``.
    """
    if not blocks:
        raise ValueError("stack_rows needs at least one block")
    n_sys = blocks[0].n_sys_qubits
    shape = blocks[0].shape
    if shape[0] != 1 << n_sys:
        raise ValueError("stack_rows requires blocks whose rows fill their system register")
    for E in blocks:
        if E.n_sys_qubits != n_sys or E.shape != shape:
            raise ValueError("stack_rows requires every block to share n_sys_qubits and shape")
    weights = [E.alpha**2 for E in blocks]
    alpha_R = float(np.sqrt(np.sum(weights)))
    PREPARE = _prepare_circuit(weights)
    s_qubits = PREPARE.num_qubits
    n_anc_shared = max(E.n_anc_qubits for E in blocks)
    n_mid = n_sys + s_qubits
    total = n_mid + n_anc_shared
    name = label or "U_R"
    selector = list(range(n_sys, n_mid))

    qc = QuantumCircuit(total, name=name)
    qc.compose(PREPARE, qubits=selector, inplace=True)
    for j, E in enumerate(blocks):
        # A block's own ancilla sits, in this circuit, past the row-select register.
        qubits = [*range(n_sys), *range(n_mid, n_mid + E.n_anc_qubits)]
        _select(qc, E.circuit, selector, j, qubits)
    return BlockEncoding(
        qc, alpha_R, n_mid, n_anc_shared, ((1 << s_qubits) * shape[0], shape[1]), name
    )


def stacked_reflection(stacked: BlockEncoding, label: str = "") -> BlockEncoding:
    r"""Return :math:`V = U_R^\dagger(2\Pi_{\rm mid} - I)U_R`, encoding :math:`2H/\alpha_R^2 - I`.

    :math:`\Pi_{\rm mid}` projects only ``stacked``'s own ancilla (its ``n_anc_qubits``, shared
    across row blocks) to zero, *not* the row-select register nested inside
    ``stacked.n_sys_qubits`` — a strictly looser projector than the one this function's own
    result exposes as its ancilla. :math:`V` is a Hermitian unitary; restricting further to
    row-select :math:`= 0` too (this result's own ``n_anc_qubits`` is exactly the row
    selector plus ``stacked``'s own ancilla) gives :math:`2H/\alpha_R^2 - I` on the true system
    register, :math:`H = R^\dagger R`.

    Parameters
    ----------
    stacked
        The output of :func:`stack_rows`.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, :math:`\alpha = 1` (the reflection is already exact at ``stacked``'s
        own :math:`\alpha_R`), ``n_sys_qubits`` the blocks' true system width, ``n_anc_qubits``
        the row-select width plus ``stacked``'s own ancilla.
    """
    n_mid = stacked.n_sys_qubits
    n_anc = stacked.n_anc_qubits
    total = n_mid + n_anc
    true_N = stacked.shape[1]
    n_sys_true = true_N.bit_length() - 1
    if (1 << n_sys_true) != true_N:
        raise ValueError(f"stacked.shape[1]={true_N} is not a power of two")
    qc = QuantumCircuit(total, name=label or "V")
    qc.compose(stacked.circuit, qubits=range(total), inplace=True)
    if n_anc > 0:
        qc.compose(_reflection_2pi_minus_i(n_anc), qubits=range(n_mid, total), inplace=True)
    qc.compose(inverse_circuit(stacked.circuit), qubits=range(total), inplace=True)
    return BlockEncoding(qc, 1.0, n_sys_true, total - n_sys_true, (true_N, true_N), label or "V")


def walk(reflection: BlockEncoding, label: str = "") -> BlockEncoding:
    r"""Return the qubitised walk :math:`W = (2\Pi_{\rm in} - I)V`.

    :math:`\Pi_{\rm in}` is ``reflection``'s own ancilla-zero projector, so this is simply
    ``reflection``'s circuit followed by a reflection on its ancilla register. On each
    two-dimensional invariant subspace of :math:`H'`, :math:`W` acts as a rotation, so
    :math:`\Pi_{\rm in} W^k \Pi_{\rm in} = T_k(H')`.

    Parameters
    ----------
    reflection
        The output of :func:`stacked_reflection` (or any square, alpha-1 block encoding of a
        Hermitian unitary).
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Same shape and register split as ``reflection``.
    """
    total = reflection.n_qubits
    qc = QuantumCircuit(total, name=label or "W")
    qc.compose(reflection.circuit, inplace=True)
    if reflection.n_anc_qubits > 0:
        qc.compose(
            _reflection_2pi_minus_i(reflection.n_anc_qubits),
            qubits=range(reflection.n_sys_qubits, total),
            inplace=True,
        )
    return BlockEncoding(
        qc, 1.0, reflection.n_sys_qubits, reflection.n_anc_qubits, reflection.shape, label or "W"
    )


def controlled_walk(
    stacked: BlockEncoding, *, n_controls: int = 1, ctrl_state: int = 0, label: str = ""
) -> QuantumCircuit:
    r"""Return the qubitised walk :math:`W` controlled on ``n_controls`` further qubits.

    :math:`W = (2\Pi_{\rm in} - I)\,U_R^\dagger(2\Pi_{\rm mid} - I)U_R` (:func:`walk` of
    :func:`stacked_reflection`) equals :math:`R_{\rm in}\,U_R^\dagger R_{\rm mid}\,U_R` with
    :math:`R = I - 2|0\rangle\langle0|` (:func:`~pihm.circuits.primitives.reflection_about_zero`),
    since the two reflections' factors of :math:`-1` cancel. Controlling :math:`W` then needs only
    the two reflections controlled: where the controls do not
    equal ``ctrl_state``, :math:`U_R^\dagger U_R = I`. :math:`U_R`'s own gates are never
    controlled, so the circuit holds nothing wider than :math:`U_R`'s widest gate besides the two
    multi-controlled :math:`Z` gates, and simulating it costs a statevector, not an operator.

    Parameters
    ----------
    stacked
        The output of :func:`stack_rows`, :math:`U_R`.
    n_controls
        Control qubits, placed above the walk's own qubits (which are ``stacked``'s).
    ctrl_state
        The value the control qubits must hold for :math:`W` to act, little-endian over them.
    label
        Circuit label.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        ``stacked.n_qubits + n_controls`` qubits: system, row select and ``stacked``'s ancilla
        (the walk's own register, as :func:`walk` orders it), then the controls.

    Raises
    ------
    ValueError
        If ``n_controls`` is not positive, or ``stacked.shape[1]`` is not a power of two.
    """
    if n_controls < 1:
        raise ValueError("controlled_walk needs at least one control qubit")
    n_walk = stacked.n_qubits
    n_mid = stacked.n_sys_qubits
    true_N = stacked.shape[1]
    n_sys = true_N.bit_length() - 1
    if (1 << n_sys) != true_N:
        raise ValueError(f"stacked.shape[1]={true_N} is not a power of two")
    controls = list(range(n_walk, n_walk + n_controls))

    def reflect(width: int) -> QuantumCircuit:
        return reflection_about_zero(width, n_controls=n_controls, ctrl_state=ctrl_state)

    qc = QuantumCircuit(n_walk + n_controls, name=label or "CW")
    qc.compose(stacked.circuit, qubits=range(n_walk), inplace=True)
    mid = list(range(n_mid, n_walk))
    if mid:
        qc.compose(reflect(len(mid)), qubits=[*mid, *controls], inplace=True)
    qc.compose(inverse_circuit(stacked.circuit), qubits=range(n_walk), inplace=True)
    ancilla = list(range(n_sys, n_walk))
    if ancilla:
        qc.compose(reflect(len(ancilla)), qubits=[*ancilla, *controls], inplace=True)
    return qc
