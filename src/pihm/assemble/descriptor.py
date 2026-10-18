r"""The descriptor form of a single field of higher order: its derivatives as blocks of the state.

The descriptor form of an equation of higher order carries the derivatives as
unknowns. Its state is a list
of blocks, each holding the coefficients of one derivative
(:attr:`~pihm.assemble.Residual.derivatives`): the field first, then along each axis the chain
:math:`\partial_a f, \dots, \partial_a^{k_a} f` up to the highest order the equation or a
constraint uses there, padded with zero blocks to a power of two, so that the block index is a
field register of :math:`n_f` qubits. The blocks are the most
significant index of the state, as axis 0 is of a field (Kronecker order). Its rows are:

* every row block of the standard form, with :math:`\mathbb{G}^j` on axis :math:`a` replaced by
  the block that carries :math:`\partial_a^j f`: the equation
  :math:`\sum_j\mathbb{M}_{a_j}p_j - \mathbb{M}_r\mathbb{D}^{(0)}p_0` (lifted exactly where the
  standard form lifts), and each constraint on its own derivative's block, with no
  :math:`\mathbb{G}`;
* the recurrences :math:`-\tilde{\mathbb{D}}p_j + \tilde{\mathbb{B}}p_{j+1} = 0` along each
  axis's chain, which hold if and only if :math:`p_{j+1} = \mathbb{G}p_j`
  (:func:`~pihm.operators.derivative.banded_factors`);
* :data:`~pihm.assemble.residual.PADDING_WEIGHT` times the identity on each padding block.

So on the states :func:`represent` builds, :math:`(b; \mathbb{G}b; \dots; 0)`, every row block
of the standard form acts as it does on :math:`b` and the recurrence and padding rows vanish, and
the two residuals have the same kernel. Their least-squares ground states
differ by the weight the derivative blocks carry, and converge to the same function.

**Block rescaling** (``rescale="block"``, the descriptor's default). The state's norm
carries every derivative, so the descriptor's gap and its field block's weight depend on how the
blocks are scaled. The block of order :math:`m_a` along each axis :math:`a` is carried divided by
:math:`\prod_a\lambda_a^{m_a}`: the residual is :math:`RS` and its state :math:`S^{-1}w`, with
:math:`S` that diagonal of block weights (:attr:`~pihm.assemble.Residual.block_weights`) and the
:math:`\lambda_a` from the equation and the resolution
(:func:`pihm.problems.scales.block_scales`). :math:`S` is invertible, so the kernel is unchanged,
and the field block's weight is 1, so the readout is too. The padding keeps weight 1: it decouples
exactly, so the physical blocks (:attr:`~pihm.assemble.Residual.physical_columns`) hold the
whole of the problem.
``rescale="none"`` keeps the unscaled residual, the ablation that shows what the rescaling does.
:func:`pihm.assemble.assemble` applies the rescaling to the blocks built here.

A first-order equation, and a first-order *system*, are carried differently, in mass form with no
derivative blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm.assemble.residual import (
    PADDING_WEIGHT,
    Block,
    Derivative,
    Readout,
    Residual,
    Truncation,
)
from pihm.assemble.standard import (
    anchor,
    constraint_block,
    lifted_axes,
    linear_term,
    power,
    source_term,
)
from pihm.operators import (
    MultiplicationMode,
    SparseArray,
    banded_factors,
    derivative_matrix,
    embed_axis,
)
from pihm.problems.reference import regular_value
from pihm.problems.spec import Problem

__all__ = ["assemble_descriptor", "descriptor_layout", "recurrence_name", "represent"]


def recurrence_name(order: int, axis: str) -> str:
    """Return the name of the recurrence row block from ``order`` to the next, along ``axis``."""
    return f"recurrence f^({order}) -> f^({order + 1}) along {axis}"


def _along(axis: int, order: int, axes: int) -> tuple[int, ...]:
    """Return the derivative of ``order`` along axis ``axis`` alone, among ``axes`` axes."""
    return tuple(order if a == axis else 0 for a in range(axes))


def descriptor_layout(problem: Problem) -> tuple[Derivative, ...]:
    """Return the derivative each block of the descriptor state carries, padding as ``None``.

    Block 0 is the field. Each axis then owns the chain of its derivatives, up to the highest
    order that the equation or a constraint takes along it, and the blocks are padded to a power
    of two.

    Raises
    ------
    NotImplementedError
        For a mixed derivative, which would need a block of its own.
    """
    axes = problem.fields[0].axes
    orders = [0] * len(axes)
    (equation,) = problem.equations
    for term in equation.terms:
        for factor in term.factors:
            if sum(1 for j in factor.derivative if j) > 1:
                raise NotImplementedError(
                    "a mixed derivative needs a descriptor block of its own, which is not "
                    "implemented"
                )
            orders = [max(k, j) for k, j in zip(orders, factor.derivative, strict=True)]
    for constraint in problem.constraints:
        index = axes.index(constraint.axis)
        orders[index] = max(orders[index], constraint.derivative)
    layout: list[Derivative] = [(0,) * len(axes)]
    for index, order in enumerate(orders):
        layout += [_along(index, j, len(axes)) for j in range(1, order + 1)]
    padded = 1 << (len(layout) - 1).bit_length()
    return (*layout, *([None] * (padded - len(layout))))


def on_block(matrix: SparseArray, index: int, count: int) -> sp.csr_array:
    """Return ``matrix`` acting on block ``index`` of a state of ``count`` blocks."""
    selector = sp.csr_array(([1.0], ([0], [index])), shape=(1, count))
    return sp.csr_array(sp.kron(selector, matrix, format="csr"))


def assemble_descriptor(
    problem: Problem, ns: tuple[int, ...], mult: MultiplicationMode, truncation: Truncation
) -> tuple[tuple[Block, ...], tuple[bool, ...], Readout, float]:
    """Return the blocks, lifts, readout and regular value of a linear descriptor residual.

    ``truncation`` is always ``"none"`` here (:func:`pihm.assemble.assemble` refuses any other);
    it keeps the signature the builders share.
    """
    axes = problem.fields[0].axes
    d = len(axes)
    (equation,) = problem.equations
    layout = descriptor_layout(problem)
    index = {derivative: i for i, derivative in enumerate(layout) if derivative is not None}
    count = len(layout)
    field = _along(0, 0, d)
    lifted = lifted_axes(problem, mult)
    out = [n + int(lift) for n, lift in zip(ns, lifted, strict=True)]
    highest = max(max(derivative) for derivative in index)
    identity = [[sp.csr_array(sp.eye_array(1 << n, format="csr"))] * (highest + 1) for n in ns]
    value = regular_value(problem)
    tau_s = anchor(problem, ns)
    size = int(np.prod([1 << n for n in ns]))

    A = sp.csr_array((int(np.prod([1 << n for n in out])), count * size))
    for term in equation.terms:
        if term.factors:
            (factor,) = term.factors
            operator = linear_term(term, axes, ns, lifted, mult, identity)
            A = sp.csr_array(A + on_block(operator, index[factor.derivative], count))
        else:
            source = source_term(term, problem, out, tau_s, value)
            A = sp.csr_array(A + on_block(source, index[field], count))
    blocks = [Block("equation", "equation", A)]

    for a, (axis, n) in enumerate(zip(axes, ns, strict=True)):
        B, D = (embed_axis(factor, ns, a) for factor in banded_factors(n))
        order = max(derivative[a] for derivative in index)
        for j in range(order):
            below, above = index[_along(a, j, d)], index[_along(a, j + 1, d)]
            rows = on_block(-D, below, count) + on_block(B, above, count)
            blocks.append(Block(recurrence_name(j, axis), "recurrence", sp.csr_array(rows)))

    padding = sp.csr_array(PADDING_WEIGHT * sp.eye_array(size, format="csr"))
    for i, derivative in enumerate(layout):
        if derivative is None:
            blocks.append(Block(f"padding block {i}", "padding", on_block(padding, i, count)))

    for constraint in problem.constraints:
        row = constraint_block(constraint, problem, ns, lifted, identity, value)
        target = index[_along(axes.index(constraint.axis), constraint.derivative, d)]
        blocks.append(Block(row.name, row.kind, on_block(row.matrix, target, count)))

    decoder = on_block(sp.eye_array(size, format="csr"), index[field], count)
    return tuple(blocks), lifted, Readout(decoder, ns, 2), value


def represent(residual: Residual, coefficients: NDArray[Any]) -> NDArray[Any]:
    r"""Return the state in ``residual``'s space that represents a field's coefficients.

    The standard form's state is the coefficients :math:`b` themselves. The descriptor form's
    is the block vector whose block for the derivative :math:`(j_0, j_1, \dots)` is
    :math:`\bigotimes_a\mathbb{G}^{j_a}\,b` divided by that block's weight
    (:attr:`~pihm.assemble.Residual.block_weights`, 1 when unscaled), with its padding blocks
    zero. It satisfies every recurrence and padding row exactly, and every other row block acts on
    it as the standard form's block of the same name acts on :math:`b`: the
    two residuals have the same kernel.

    Parameters
    ----------
    residual
        A residual of a linear problem with one field, in either regime.
    coefficients
        The field's normalised coefficients, in Kronecker order on ``residual.qubits``.

    Returns
    -------
    numpy.ndarray
        The state, of length ``residual.dimension``.

    Raises
    ------
    ValueError
        For a doubled space, whose state :math:`b\otimes b` is not linear in :math:`b`, and for a
        system, whose state needs every component's coefficients
        (:func:`pihm.assemble.systems.system_state`).

    Examples
    --------
    >>> from pihm.assemble import assemble
    >>> from pihm.problems import ladder
    >>> residual = assemble(ladder.problem("R1"), 2, regime="descriptor", layout="chain")
    >>> represent(residual, np.array([1.0, 0.0, 0.0, 0.0]))  # f = 1/2, f' = 0
    array([1., 0., 0., 0., 0., 0., 0., 0.])
    """
    if residual.nonlinearity == "doubled":
        raise ValueError("a doubled space's state b (x) b is not linear in b")
    if residual.is_system:
        raise ValueError("a system's state carries every component: use systems.system_state")
    b = np.asarray(coefficients)
    blocks = []
    for derivative, weight in zip(residual.derivatives, residual.block_weights, strict=True):
        block = np.zeros_like(b) if derivative is None else b
        for axis, order in enumerate(derivative or ()):
            if order:
                operator = power(derivative_matrix(residual.qubits[axis]), order)
                block = np.asarray(embed_axis(operator, residual.qubits, axis) @ block)
        blocks.append(block / weight if weight != 1.0 else block)
    return np.concatenate(blocks)
