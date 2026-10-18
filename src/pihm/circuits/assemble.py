r"""Compile a problem into circuit block encodings, in either regime.

:func:`encode_standard_linear` mirrors :func:`pihm.assemble.assemble`'s own term-walking logic
(:math:`\mathbb{A} = \sum_j c_j\mathbb{M}_{m_j}\mathbb{G}^j`, plus the invariant rows
:math:`\mathbb{B}(x^*)\mathbb{G}^j`), but builds
:class:`~pihm.circuits.primitives.BlockEncoding`\ s instead of sparse matrices, so a rung's
Pillar 1 record comes from the *same* problem specification as its classical one.

Scope: one field, whose residual is linear in the state. That covers every rung of the ladder,
generically rather than by a hand-built circuit per rung: R1 and every panel of R2 (Fig. 3),
R3 (Fig. 4) and R4 (Fig. 5) on one axis with invariant constraints; R5 (Fig. 6) on two square
axes, each operator a Kronecker product, with datum slices; and R6 (Fig. 7), quadratic, in the
paper's doubled space :math:`\Psi=\psi\otimes\psi`, its terms folded by :math:`\mathbb{N}_1` and
:math:`\mathbb{N}_x` (:func:`~pihm.circuits.folds.fold_encoding`).

The axis is lifted exactly when :func:`pihm.assemble.assemble` lifts it (a non-constant
multiplier, or a source). A lifted term is :math:`c\,\mathbb{M}_m\mathbb{G}^j = c\,\bigl(\sum_k a_k
\mathbb{M}_x^k\bigr)\mathbb{M}_1\mathbb{G}^j` on :math:`n+1` qubits
(:func:`~pihm.circuits.multiplication.polynomial_encoding` after
:func:`~pihm.circuits.multiplication.embed`), a constant multiplier's :math:`\mathbb{M}_c` being
:math:`c\,\mathbb{M}_1`; a lifted invariant row is :math:`\mathbb{B}_{n+1}(x^*)\mathbb{M}_1
\mathbb{G}^j`; a source :math:`r` is :math:`-\mathbb{M}_r\mathbb{D}^{(0)}(x_s)`
(:func:`~pihm.circuits.sources.source_encoding`). Powers :math:`\mathbb{G}^j` are
:func:`~pihm.circuits.compose.product`\ s, which share one ancilla register, so
the circuit's width grows by one flag qubit per factor rather than by the factor's ancilla.
Resource counts (:math:`\alpha`, ancilla, gate counts) are read directly off the circuit at every
:math:`n`, needing no simulation. :math:`\mathbb{G}` is the structured exact encoding, or, with
``encoder="published"``, the paper's own (Fig. 9, approximate;
:func:`~pihm.circuits.standard.published_derivative_encoding`), whose resources tier T3 estimates;
the paper's :math:`\mathbb{B}` and :math:`\mathbb{D}^{(0)}` (Figs 2 and 8)
are the rows' atoms in both.

:func:`encode_descriptor_linear` builds the descriptor form's stacked residual for every linear
problem :func:`pihm.assemble.assemble` assembles in that form (R1 to R5), block-rescaled as the
residual is:

* **the system rows** (the equation, the recurrences and the padding) are one row block, the term
  IR (:func:`pihm.ir.system_ir`) compiled by :func:`~pihm.circuits.lcu.compile_system`: its bands
  a factored LCU, and a source the rank-one :math:`|\hat r\rangle\langle\tau(x_s)|` from the
  field block to the equation's row;
* **each constraint** is a row block of its own: the standard form's atom with
  :math:`\mathbb{G}^j` replaced by the block that carries :math:`\partial^jf`, which it reads
  (:func:`~pihm.circuits.lcu.field_placement`), its :math:`\alpha` times that block's weight
  :math:`\lambda^j`.

Its rows keep a lifted axis's extra qubit above the field register
(:func:`~pihm.circuits.lcu.circuit_rows`), so the state is the reflection's system register.

:func:`encode_linear` takes the regime and calls the one that builds it, or for a first-order
system, several fields or a Carleman lift, :func:`pihm.circuits.systems.encode_system`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from pihm import basis
from pihm.assemble import Regime, Rescale, Residual, assemble, is_system
from pihm.circuits.compose import (
    embed_system,
    lcu_sum,
    product,
    real_state_preparation,
    stack_rows,
    stacked_reflection,
)
from pihm.circuits.folds import fold_encoding
from pihm.circuits.lcu import Layout, compile_system, field_placement, place
from pihm.circuits.multiplication import embed, polynomial_encoding
from pihm.circuits.primitives import BlockEncoding, identity_encoding
from pihm.circuits.sources import source_encoding
from pihm.circuits.standard import (
    DERIVATIVE_ENCODERS,
    StandardEncoder,
    collapse_encoding,
    zero_encoding,
)
from pihm.circuits.systems import encode_system
from pihm.ir import system_ir
from pihm.problems.expressions import compile_expression
from pihm.problems.reference import regular_value
from pihm.problems.spec import Constraint, Problem

__all__ = [
    "descriptor_blocks",
    "encode_descriptor_linear",
    "encode_linear",
    "encode_standard_linear",
]


def _check_scope(problem: Problem) -> None:
    """Refuse a problem outside the compiler's circuit coverage, with the reason."""
    if len(problem.fields) != 1:
        raise NotImplementedError("encode_standard_linear covers one field")
    (equation,) = problem.equations
    several = len(problem.fields[0].axes) > 1
    if problem.degree > 2 or (problem.degree == 2 and several):
        raise NotImplementedError(
            "encode_standard_linear covers linear equations, and quadratic ones on one axis"
        )
    if problem.degree == 2:
        for term in equation.terms:
            if term.source is not None:
                raise NotImplementedError("the doubled space folds polynomial terms only")
            if term.multiplier is not None and term.multiplier.degree > 1:
                raise NotImplementedError("the doubled space folds multipliers of degree <= 1")
        if any(c.kind != "zero" for c in problem.constraints):
            raise NotImplementedError("the doubled space takes invariant constraints only")
        return
    for term in equation.terms:
        lifted = not term.factors or (
            term.multiplier is not None and not term.multiplier.is_constant
        )
        if several and lifted:
            raise NotImplementedError("encode_standard_linear lifts one axis only")
    for constraint in problem.constraints:
        if constraint.kind == "datum" and several and constraint.derivative == 0:
            continue
        if constraint.kind != "zero":
            raise NotImplementedError(
                "encode_standard_linear covers invariant constraints, and datum slices of a "
                "field on several axes"
            )


def _powers(n: int, encoder: StandardEncoder) -> Callable[[int], BlockEncoding]:
    r"""Return :math:`j\mapsto\mathbb{G}^j` on ``n`` qubits, each power built once."""
    G = DERIVATIVE_ENCODERS[encoder](n)
    powers: dict[int, BlockEncoding] = {0: identity_encoding(n), 1: G}

    def power(order: int) -> BlockEncoding:
        if order not in powers:
            powers[order] = product(G, power(order - 1))
        return powers[order]

    return power


def _one_axis_blocks(problem: Problem, n: int, encoder: StandardEncoder) -> list[BlockEncoding]:
    """Return the row blocks of a one-axis problem, lifted where the paper lifts."""
    (equation,) = problem.equations
    lifted = any(
        not t.factors or (t.multiplier is not None and not t.multiplier.is_constant)
        for t in equation.terms
    )
    power = _powers(n, encoder)
    terms: list[BlockEncoding] = []
    weights: list[complex] = []
    for term in equation.terms:
        if not term.factors:
            terms.append(source_encoding(term, problem, n))
            weights.append(term.coefficient)
            continue
        (factor,) = term.factors
        (order,) = factor.derivative
        multiplier = term.multiplier
        coefficient = term.coefficient
        if multiplier is not None and multiplier.is_constant:
            coefficient *= multiplier.coefficients[0]
            multiplier = None
        if not lifted:
            terms.append(power(order))
        elif multiplier is None:
            terms.append(embed(power(order)))
        else:
            terms.append(
                product(polynomial_encoding(multiplier.coefficients, n + 1), embed(power(order)))
            )
        weights.append(coefficient)
    blocks = [lcu_sum(terms, weights, label="A")]
    for constraint in problem.constraints:
        slope = power(constraint.derivative)
        if lifted:
            blocks.append(product(zero_encoding(constraint.at, n + 1), embed(slope)))
        else:
            blocks.append(product(zero_encoding(constraint.at, n), slope))
    return blocks


def _axes_blocks(
    problem: Problem, ns: tuple[int, ...], encoder: StandardEncoder
) -> list[BlockEncoding]:
    r"""Return the row blocks of a problem on several axes, each operator a Kronecker product.

    Axis :math:`a` holds the qubits below the axes before it (Kronecker order, the first axis
    most significant), and a Kronecker product is a :func:`~pihm.circuits.compose.product` of
    :func:`~pihm.circuits.compose.embed_system`\ s, sharing one ancilla register.
    """
    axes = problem.fields[0].axes
    total = sum(ns)
    positions = [list(range(sum(ns[a + 1 :]), sum(ns[a:]))) for a in range(len(ns))]
    power = [_powers(n, encoder) for n in ns]

    def on_axes(parts: dict[int, BlockEncoding]) -> BlockEncoding:
        embedded = [embed_system(part, positions[a], total) for a, part in sorted(parts.items())]
        if not embedded:
            return identity_encoding(total)
        result = embedded[-1]
        for part in reversed(embedded[:-1]):
            result = product(part, result)
        return result

    (equation,) = problem.equations
    terms: list[BlockEncoding] = []
    weights: list[complex] = []
    for term in equation.terms:
        (factor,) = term.factors
        coefficient = term.coefficient
        if term.multiplier is not None:
            coefficient *= term.multiplier.coefficients[0]
        orders = {a: power[a](j) for a, j in enumerate(factor.derivative) if j}
        terms.append(on_axes(orders))
        weights.append(coefficient)
    blocks = [lcu_sum(terms, weights, label="A")]
    for constraint in problem.constraints:
        if constraint.kind == "datum":
            blocks.append(_datum_encoding(problem, constraint, ns, positions, on_axes))
            continue
        a = axes.index(constraint.axis)
        row = product(zero_encoding(constraint.at, ns[a]), power[a](constraint.derivative))
        blocks.append(on_axes({a: row}))
    return blocks


def _datum_encoding(
    problem: Problem,
    constraint: Constraint,
    ns: tuple[int, ...],
    positions: list[list[int]],
    on_axes: Callable[[dict[int, BlockEncoding]], BlockEncoding],
) -> BlockEncoding:
    r"""Return the datum slice :math:`\mathbb{S}_a(x^*)-\hat g\,\tau(x_s)^{\top}/v`.

    Both parts live on :math:`|0\rangle` of axis :math:`a`: the slice is :math:`\mathbb{B}(x^*)`
    there, and the injection :math:`\sqrt{N_a}|0\rangle|g\rangle\langle\tau(x_s)|/v` is
    :math:`(\|g\|/\sqrt{N_o})\,\mathbb{D}^{(0)}_a(x_s)\otimes S_g\mathbb{B}_o(x_s)`, with
    :math:`\mathbb{B}_o` on every other axis, :math:`N_o` their joint size, and :math:`S_g` the
    rotation tree preparing the datum's coefficients :math:`g/\|g\|` on them
    (:func:`~pihm.circuits.compose.real_state_preparation`), as
    :func:`pihm.operators.constraints.datum_rows` builds it.
    """
    constraint_axes = problem.fields[0].axes
    a = constraint_axes.index(constraint.axis)
    assert constraint.datum is not None  # a datum constraint carries its function
    others = [b for b in range(len(ns)) if b != a]
    g = basis.fit(
        compile_expression(constraint.datum, [constraint_axes[b] for b in others]),
        [ns[b] for b in others],
    )
    if np.iscomplexobj(g) and np.any(g.imag != 0):
        raise NotImplementedError("a datum slice loads a real function only")
    g = np.real(g)
    anchor = problem.regular.at
    rows = {b: zero_encoding(anchor[b], ns[b]) for b in others}
    rows[a] = collapse_encoding(anchor[a], regular_value(problem), ns[a])
    joint = [q for b in reversed(others) for q in positions[b]]
    size = g.size
    loader = BlockEncoding(
        real_state_preparation(g, "S_g"), 1.0, len(joint), 0, (size, size), "S_g"
    )
    injection = product(embed_system(loader, joint, sum(ns)), on_axes(rows))
    scale = float(np.linalg.norm(g)) / np.sqrt(size)
    injection = BlockEncoding(
        injection.circuit,
        injection.alpha * scale,
        injection.n_sys_qubits,
        injection.n_anc_qubits,
        injection.shape,
        "g·tau(x_s)/v",
    )
    slice_row = on_axes({a: zero_encoding(constraint.at, ns[a])})
    return lcu_sum([slice_row, injection], [1.0, -1.0], label="datum")


def _doubled_blocks(problem: Problem, n: int, encoder: StandardEncoder) -> list[BlockEncoding]:
    r"""Return the row blocks of a quadratic problem in the paper's doubled space (Eqs 30 to 34).

    The state is :math:`\Psi=\psi\otimes\psi` on two :math:`n`-qubit copies, the first the
    higher. A term with :math:`k\le2` factors is :math:`c\,\mathbb{N}_m(\mathbb{D}^{(0)\otimes
    (2-k)}\otimes\mathbb{G}^{j_1}\otimes\cdots)`, :math:`\mathbb{N}_m=a_0\mathbb{N}_1+a_1
    \mathbb{N}_x` for the multiplier :math:`a_0+a_1x` (:func:`~pihm.circuits.folds.fold_encoding`),
    and an invariant row is :math:`\mathbb{N}_1(\mathbb{D}^{(0)}(x_s)\otimes\mathbb{B}(x^*)
    \mathbb{G}^j)`, as :func:`pihm.assemble.assemble` builds them.
    """
    power = _powers(n, encoder)
    collapse = collapse_encoding(problem.regular.at[0], regular_value(problem), n)
    folds = {weight: fold_encoding(n, weight) for weight in ("none", "x")}

    def copies(first: BlockEncoding, second: BlockEncoding) -> BlockEncoding:
        high = embed_system(first, list(range(n, 2 * n)), 2 * n)
        return product(high, embed_system(second, list(range(n)), 2 * n))

    (equation,) = problem.equations
    terms: list[BlockEncoding] = []
    weights: list[complex] = []
    for term in equation.terms:
        factors = [power(f.derivative[0]) for f in term.factors]
        pair = copies(*([collapse] * (2 - len(factors)) + factors))
        coefficients = (1.0,) if term.multiplier is None else term.multiplier.coefficients
        for weight, a in zip(("none", "x"), [*coefficients, 0.0][:2], strict=True):
            if a != 0.0:
                terms.append(product(folds[weight], pair))
                weights.append(term.coefficient * a)
    blocks = [lcu_sum(terms, weights, label="A")]
    for constraint in problem.constraints:
        row = product(zero_encoding(constraint.at, n), power(constraint.derivative))
        blocks.append(product(folds["none"], copies(collapse, row)))
    return blocks


def encode_standard_linear(
    problem: Problem, n: int | Sequence[int], *, encoder: StandardEncoder = "structured"
) -> tuple[BlockEncoding, BlockEncoding]:
    r"""Return the stacked residual :math:`U_R` and its reflection :math:`V` for ``problem``.

    Parameters
    ----------
    problem
        A one-field problem: linear on one axis with invariant constraints (R1, or any panel of
        R2, R3 or R4), linear on several square axes with invariant and datum constraints (R5),
        or quadratic on one axis, in the doubled space (R6).
    n
        Qubits per axis (one number for every axis, or one per axis).
    encoder
        The encoding of :math:`\mathbb{G}` (:data:`~pihm.circuits.standard.DERIVATIVE_ENCODERS`).
        The published one is approximate, so its reflection only approximates
        :math:`2H/\alpha_R^2 - I`.

    Returns
    -------
    stacked, V
        ``stacked`` is :func:`~pihm.circuits.compose.stack_rows`'s own result (useful for its
        :math:`\alpha_R`); ``V`` is :func:`~pihm.circuits.compose.stacked_reflection`'s, encoding
        :math:`2H/\alpha_R^2 - I`.

    Raises
    ------
    NotImplementedError
        If ``problem`` is outside the compiler's coverage (see the module docstring).
    """
    _check_scope(problem)
    count = len(problem.fields[0].axes)
    ns = (n,) * count if isinstance(n, int) else tuple(n)
    if len(ns) != count:
        raise ValueError(f"{problem.key} has {count} axes, got n = {ns}")
    if problem.degree == 2:
        blocks = _doubled_blocks(problem, ns[0], encoder)
    elif count == 1:
        blocks = _one_axis_blocks(problem, ns[0], encoder)
    else:
        blocks = _axes_blocks(problem, ns, encoder)
    stacked = stack_rows(blocks)
    return stacked, stacked_reflection(stacked)


def _constraint_atom(
    residual: Residual, constraint: Constraint, layout: Layout
) -> tuple[BlockEncoding, list[int]]:
    """Return a descriptor constraint's atom on the axes, and the system qubits it acts on."""
    problem = residual.problem
    ns = residual.qubits
    a = problem.fields[0].axes.index(constraint.axis)
    if constraint.kind == "zero":
        if residual.lifted[a]:
            lifted = product(
                zero_encoding(constraint.at, ns[a] + 1), embed(identity_encoding(ns[a]))
            )
            return lifted, layout.output(a)
        return zero_encoding(constraint.at, ns[a]), layout.data(a)
    if constraint.kind == "datum" and constraint.derivative == 0 and not any(residual.lifted):
        total = sum(ns)
        positions = [layout.data(b) for b in range(len(ns))]

        def on_axes(parts: dict[int, BlockEncoding]) -> BlockEncoding:
            embedded = [
                embed_system(part, positions[b], total) for b, part in sorted(parts.items())
            ]
            result = embedded[-1]
            for part in reversed(embedded[:-1]):
                result = product(part, result)
            return result

        return _datum_encoding(problem, constraint, ns, positions, on_axes), list(range(total))
    raise NotImplementedError(
        "the descriptor's constraints are invariant rows, and datum slices of the field on "
        "axes that are not lifted"
    )


def descriptor_blocks(residual: Residual) -> list[BlockEncoding]:
    """Return the row blocks of a linear descriptor residual (the module documentation).

    Parameters
    ----------
    residual
        From :func:`pihm.assemble.assemble` with ``regime="descriptor"``, rescaled or not.

    Returns
    -------
    list of BlockEncoding
        The system rows, then one block per constraint, each on the system register of
        :class:`~pihm.circuits.lcu.Layout`, in :func:`~pihm.circuits.lcu.circuit_rows`' order.

    Raises
    ------
    NotImplementedError
        For a constraint the atoms do not cover.
    """
    problem = residual.problem
    ir = system_ir(residual)
    layout = Layout.of(ir)
    axes = problem.fields[0].axes
    index = {d: i for i, d in enumerate(residual.derivatives) if d is not None}
    blocks = [compile_system(ir, label="A_sys")]
    for constraint in problem.constraints:
        a = axes.index(constraint.axis)
        target = index[tuple(constraint.derivative if b == a else 0 for b in range(len(axes)))]
        atom, qubits = _constraint_atom(residual, constraint, layout)
        placement = field_placement(0, target, layout.field_qubits)
        row = place([(placement, layout.field()), (atom, qubits)], layout, label=constraint.kind)
        blocks.append(dataclasses.replace(row, alpha=row.alpha * residual.block_weights[target]))
    return blocks


def encode_descriptor_linear(
    problem: Problem, n: int | Sequence[int], *, rescale: Rescale | None = None
) -> tuple[BlockEncoding, BlockEncoding]:
    r"""Return the descriptor form's stacked residual :math:`U_R` and its reflection :math:`V`.

    Parameters
    ----------
    problem
        A linear one-field problem the descriptor form assembles (R1 to R5).
    n
        Qubits per axis (one number for every axis, or one per axis).
    rescale
        The block rescaling, as :func:`pihm.assemble.assemble` takes it: by default
        ``"block"``.

    Returns
    -------
    stacked, V
        As :func:`encode_standard_linear` returns them; ``V`` encodes :math:`2H/\alpha_R^2 - I`
        for the rescaled :math:`H = (RS)^\top RS` on the state's register.
    """
    count = len(problem.fields[0].axes)
    ns = (n,) * count if isinstance(n, int) else tuple(n)
    residual = assemble(problem, ns, regime="descriptor", rescale=rescale)
    stacked = stack_rows(descriptor_blocks(residual))
    return stacked, stacked_reflection(stacked)


def encode_linear(
    problem: Problem,
    n: int | Sequence[int],
    *,
    regime: Regime = "standard",
    rescale: Rescale | None = None,
    encoder: StandardEncoder = "structured",
    carleman: int | None = None,
) -> tuple[BlockEncoding, BlockEncoding]:
    r"""Return the regime's stacked residual and its reflection, as Pillar 1 verifies them.

    :func:`encode_standard_linear`, whose ``encoder`` this passes on, or
    :func:`encode_descriptor_linear`, whose ``rescale`` it passes on; the standard form has no
    block rescaling. A first-order system (:func:`pihm.assemble.is_system`), a Carleman lift's
    among them, is :func:`pihm.circuits.systems.encode_system`'s.

    Raises
    ------
    ValueError
        For the descriptor form with an encoder of :math:`\mathbb{G}`, which it does not have.
    """
    if regime == "descriptor" and encoder != "structured":
        raise ValueError(f"the descriptor form has no G to encode, so no {encoder!r} encoder")
    if is_system(problem, regime, carleman):
        return encode_system(
            problem, n, regime=regime, rescale=rescale, carleman=carleman, encoder=encoder
        )
    if regime == "descriptor":
        return encode_descriptor_linear(problem, n, rescale=rescale)
    return encode_standard_linear(problem, n, encoder=encoder)
