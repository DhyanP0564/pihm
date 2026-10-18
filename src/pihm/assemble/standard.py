r"""The standard form of a single field, exactly as the paper builds it (:cite:`wu2025pihm`).

The paper's effective Hamiltonian is
:math:`\mathbb{H} = \mathcal{T}(\mathbb{A}) + \sum_i \mathcal{T}(\mathbb{B}_i)`, the stacked
residual :math:`R = [\mathbb{A}; \mathbb{B}_1; \dots]` (the paper's Eqs 15 to 34):

* A linear term :math:`c\,m(x)\,\partial^\alpha f` becomes :math:`c\,\mathbb{M}_m\mathbb{G}^\alpha`,
  axis by axis (Kronecker order). Terms with a non-constant multiplier, and sources, need the
  exact product on one extra qubit, so under ``mult="lift"`` such an axis is **lifted**: every
  term on it goes through :math:`\mathbb{M}_m` (:math:`\mathbb{M}_1` for a constant), which is
  the paper's Figs 4 and 5. Axes that need no lift stay square, as in Figs 3 and 6.
  ``mult="square"`` never lifts, and uses the Galerkin truncation instead.
* A source :math:`r(x)` (a term without factors) is injected through the regular constraint,
  :math:`\mathbb{M}_r\mathbb{D}^{(0)}(x_s) = |\hat r\rangle\langle\tau(x_s)|/f(x_s)`, with
  :math:`\hat r` the coefficients of :math:`r` on the lifted register. For a Maclaurin source this
  is the paper's :math:`\sum_p c_p\mathbb{M}_{x^p}\mathbb{D}^{(0)}(x_s)` (Eq. 14) exactly.
* An invariant constraint :math:`\partial_a^j f = 0` on the slice :math:`x_a = x^*` is the row
  block :math:`\mathbb{B}_n(x^*)\mathbb{G}^j` on axis :math:`a` and the identity elsewhere, or
  :math:`\mathbb{B}_{n+1}(x^*)\mathbb{M}_1\mathbb{G}^j` on a lifted axis (Eqs 9, 10, 19, 20, 24
  and 25). A datum constraint is :func:`pihm.operators.constraints.datum_rows`, anchored at
  :math:`x_s`.
* ``nonlinearity="doubled"``: the state is :math:`\Psi = \psi\otimes\psi` (Eqs 30 to 34). A term
  with :math:`k \le 2` factors becomes :math:`c\,\mathbb{N}_m(\mathbb{D}^{(0)\otimes(2-k)} \otimes
  \mathbb{G}^{j_1} \otimes \dots)`, with :math:`\mathbb{N}_1` or :math:`\mathbb{N}_x` for the
  multiplier, and an invariant constraint becomes
  :math:`\mathbb{N}_1(\mathbb{D}^{(0)}(x_s) \otimes \mathbb{B}_n(x^*)\mathbb{G}^j)`.

``truncation="tau"`` is an ablation (R2 only): the top :math:`k` rows of the
equation block of a square, one-axis, order-:math:`k` problem are dropped, which leaves an exact
one-dimensional kernel (the Chebyshev tau method).

The helpers that build one term or one constraint are shared with the descriptor form
(:mod:`pihm.assemble.descriptor`), which places the same rows on its derivative blocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm import basis
from pihm.assemble.residual import Block, Readout, Truncation
from pihm.operators import (
    MultiplicationMode,
    SparseArray,
    collapse_row,
    datum_rows,
    derivative_matrix,
    embedding,
    fold,
    kron_all,
    multiplication,
    zero_row,
)
from pihm.problems.expressions import compile_expression
from pihm.problems.functions import polynomial_coefficients, source_coefficients
from pihm.problems.reference import regular_value
from pihm.problems.spec import Constraint, Problem, Term

__all__ = ["assemble_doubled", "assemble_linear"]


def power(matrix: SparseArray, exponent: int) -> sp.csr_array:
    """Return a sparse matrix power (the identity for exponent 0)."""
    result = sp.csr_array(sp.eye_array(matrix.shape[0], format="csr"))
    for _ in range(exponent):
        result = sp.csr_array(result @ matrix)
    return result


def derivative_powers(ns: Sequence[int], order: int) -> list[list[sp.csr_array]]:
    r"""Return the powers :math:`\mathbb{G}^0 .. \mathbb{G}^{k}` for each axis."""
    return [[power(derivative_matrix(n), j) for j in range(order + 1)] for n in ns]


def _constant_coefficients(n: int) -> NDArray[np.float64]:
    """Return the normalised coefficients of the constant function 1 on :math:`n` qubits."""
    values = np.zeros(1 << n)
    values[0] = np.sqrt(float(1 << n))
    return values


def lifted_axes(problem: Problem, mult: MultiplicationMode) -> tuple[bool, ...]:
    """Return, per axis, whether the paper lifts it: a non-constant multiplier or a source."""
    axes = problem.fields[0].axes
    if mult == "square":
        return (False,) * len(axes)
    (equation,) = problem.equations
    has_source = any(not term.factors for term in equation.terms)
    return tuple(
        has_source
        or any(
            t.multiplier is not None and t.multiplier.axis == axis and not t.multiplier.is_constant
            for t in equation.terms
        )
        for axis in axes
    )


def linear_term(
    term: Term,
    axes: tuple[str, ...],
    ns: Sequence[int],
    lifted: tuple[bool, ...],
    mult: MultiplicationMode,
    powers: list[list[sp.csr_array]],
) -> sp.csr_array:
    r"""Return :math:`c\,\bigotimes_a \mathbb{M}_a \mathbb{G}^{j_a}` for a one-factor term."""
    (factor,) = term.factors
    parts: list[SparseArray] = []
    for index, (axis, n) in enumerate(zip(axes, ns, strict=True)):
        operator = powers[index][factor.derivative[index]]
        polynomial = term.multiplier if term.multiplier and term.multiplier.axis == axis else None
        coefficients = (1.0,) if polynomial is None else polynomial.coefficients
        if lifted[index]:
            operator = sp.csr_array(multiplication(coefficients, n, mode="lift") @ operator)
        elif polynomial is not None and not polynomial.is_constant:
            operator = sp.csr_array(multiplication(coefficients, n, mode=mult) @ operator)
        elif polynomial is not None:
            operator = sp.csr_array(coefficients[0] * operator)
        parts.append(operator)
    return sp.csr_array(term.coefficient * kron_all(parts))


def source_term(
    term: Term,
    problem: Problem,
    out: Sequence[int],
    anchor: NDArray[Any],
    value: float,
) -> sp.csr_array:
    r"""Return :math:`c\,|\hat r\rangle\langle\tau(x_s)|/u(x_s)` on the output registers."""
    axes = problem.fields[0].axes
    if term.source is not None:
        target = source_coefficients(term.source, [a.name for a in problem.axes], out)
    else:
        parts = []
        for axis, n in zip(axes, out, strict=True):
            polynomial = (
                term.multiplier if term.multiplier and term.multiplier.axis == axis else None
            )
            parts.append(
                _constant_coefficients(n)
                if polynomial is None
                else polynomial_coefficients(polynomial, n)
            )
        target = parts[0]
        for part in parts[1:]:
            target = np.kron(target, part)
    rows = np.flatnonzero(target)
    data = np.outer(target[rows], anchor * (term.coefficient / value)).ravel()
    return sp.csr_array(
        (data, (np.repeat(rows, anchor.size), np.tile(np.arange(anchor.size), rows.size))),
        shape=(target.size, anchor.size),
    )


def constraint_block(
    constraint: Constraint,
    problem: Problem,
    ns: Sequence[int],
    lifted: tuple[bool, ...],
    powers: list[list[sp.csr_array]],
    value: float,
) -> Block:
    """Return the row block of one constraint, in the linear standard form."""
    axes = problem.fields[0].axes
    index = axes.index(constraint.axis)
    name = (
        f"{constraint.kind} f^({constraint.derivative}) on {constraint.axis} = {constraint.at:.10g}"
    )
    if constraint.kind == "datum":
        assert constraint.datum is not None
        others = [(a, n) for a, n in zip(axes, ns, strict=True) if a != constraint.axis]
        g = compile_expression(constraint.datum, [a for a, _ in others])
        g_hat = basis.fit(g, [n for _, n in others]) if others else np.ones(1)
        rows = datum_rows(list(ns), index, constraint.at, g_hat, problem.regular.at, value)
        return Block(name, "constraint", rows)
    n = ns[index]
    slope = powers[index][constraint.derivative]
    if lifted[index]:
        row = sp.csr_array(zero_row(constraint.at, n + 1) @ embedding(n) @ slope)
    else:
        row = sp.csr_array(zero_row(constraint.at, n) @ slope)
    parts: list[SparseArray] = [
        row if i == index else sp.eye_array(1 << m, format="csr") for i, m in enumerate(ns)
    ]
    return Block(name, "constraint", kron_all(parts))


def anchor(problem: Problem, ns: Sequence[int]) -> NDArray[Any]:
    r"""Return the raw feature vector :math:`\tau(x_s)` on the input registers (Kronecker order)."""
    vector: NDArray[Any] = np.ones(1)
    for coordinate, n in zip(problem.regular.at, ns, strict=True):
        vector = np.kron(vector, basis.tau(coordinate, n))
    return vector


def assemble_linear(
    problem: Problem, ns: tuple[int, ...], mult: MultiplicationMode, truncation: Truncation
) -> tuple[tuple[Block, ...], tuple[bool, ...], Readout, float]:
    """Return the blocks, lifts, readout and regular value of a linear standard-form residual."""
    axes = problem.fields[0].axes
    (equation,) = problem.equations
    lifted = lifted_axes(problem, mult)
    out = [n + int(lift) for n, lift in zip(ns, lifted, strict=True)]
    order = max([problem.order, *(c.derivative for c in problem.constraints)])
    powers = derivative_powers(ns, order)
    value = regular_value(problem)
    tau_s = anchor(problem, ns)
    size_in = int(np.prod([1 << n for n in ns]))
    size_out = int(np.prod([1 << n for n in out]))
    A = sp.csr_array((size_out, size_in))
    for term in equation.terms:
        if term.factors:
            A = sp.csr_array(A + linear_term(term, axes, ns, lifted, mult, powers))
        else:
            A = sp.csr_array(A + source_term(term, problem, out, tau_s, value))
    if truncation == "tau":
        if len(ns) != 1 or any(lifted):
            raise ValueError("the tau truncation is defined for square one-axis problems (R2)")
        A = sp.csr_array(A[: A.shape[0] - problem.order])
    blocks = (
        Block("equation", "equation", A),
        *(constraint_block(c, problem, ns, lifted, powers, value) for c in problem.constraints),
    )
    readout = Readout(sp.csr_array(sp.eye_array(size_in, format="csr")), ns, 2)
    return blocks, lifted, readout, value


def assemble_doubled(
    problem: Problem, ns: tuple[int, ...], mult: MultiplicationMode, truncation: Truncation
) -> tuple[tuple[Block, ...], tuple[bool, ...], Readout, float]:
    """Return the blocks, lifts, readout and regular value in the paper's doubled space."""
    if len(ns) != 1:
        raise NotImplementedError("the doubled space is implemented on one axis")
    if mult != "lift" or truncation != "none":
        raise ValueError("the doubled space takes mult='lift' and truncation='none' only")
    (n,) = ns
    (equation,) = problem.equations
    value = regular_value(problem)
    collapse = collapse_row(problem.regular.at[0], value, n)
    powers = derivative_powers(
        ns, max([problem.order, *(c.derivative for c in problem.constraints)])
    )[0]
    folds = (fold(2, n), fold(2, n, weight="x"))
    A = sp.csr_array(((2 << n), (1 << n) ** 2))
    for term in equation.terms:
        if term.source is not None:
            raise NotImplementedError(
                "non-polynomial sources are not implemented in a doubled space"
            )
        coefficients = (1.0,) if term.multiplier is None else term.multiplier.coefficients
        if term.multiplier is not None and term.multiplier.degree > 1:
            raise NotImplementedError("the doubled space folds multipliers of degree <= 1 only")
        weights = [*coefficients[:2], 0.0][:2]
        folded = sp.csr_array(weights[0] * folds[0] + weights[1] * folds[1])
        copies = [collapse] * (2 - term.degree) + [powers[f.derivative[0]] for f in term.factors]
        A = sp.csr_array(
            A + term.coefficient * (folded @ sp.kron(copies[0], copies[1], format="csr"))
        )
    blocks = [Block("equation", "equation", A)]
    for constraint in problem.constraints:
        if constraint.kind != "zero":
            raise NotImplementedError("only invariant constraints exist in the doubled space")
        row = sp.csr_array(zero_row(constraint.at, n) @ powers[constraint.derivative])
        name = f"zero f^({constraint.derivative}) on x = {constraint.at:.10g} (doubled)"
        blocks.append(Block(name, "constraint", sp.csr_array(folds[0] @ sp.kron(collapse, row))))
    identity = sp.eye_array(1 << n, format="csr")
    readout = Readout(sp.csr_array(folds[0] @ sp.kron(collapse, identity)), (n + 1,), 1)
    return tuple(blocks), (True,), readout, value
