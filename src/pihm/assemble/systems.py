r"""First-order systems in both forms: several fields, or a nonlinear field lifted by Carleman.

A system here is :math:`u_i' = \sum_p (F_p u^{\otimes p})_i` on one axis, one equation per field,
each holding the first derivative of its own field with a constant coefficient and otherwise only
constant multiples of products of the fields (:func:`vector_field`). Its components are the
monomials of the fields up to the Carleman order :math:`K` (:func:`pihm.carleman.monomials`; the
fields alone for a linear system, :math:`K = 1`), and it is the linear system
:math:`z' = \mathbb{A}z` with :math:`\mathbb{A}` the Carleman matrix
(:func:`pihm.carleman.carleman_matrix`; :math:`F_1` itself when linear). Rungs R7 (coupled
fields) and R8 (Carleman) are systems, and in the descriptor form so is R1, a single first-order
equation.

**The state** is one block per component, the most significant index, padded to a power of two
(:attr:`~pihm.assemble.residual.Residual.monomials`). Both forms carry the same state, so the
initial state and the readout are the same in both.

**The rows**, with components :math:`c` and the axis's Chebyshev operators:

* the equations: in the standard form :math:`I \otimes \mathbb{G} - \mathbb{A} \otimes I`; in the
  descriptor form the same rows multiplied by :math:`I \otimes \tilde{\mathbb{B}}`, which is
  banded and invertible, so that with :math:`\tilde{\mathbb{B}}\mathbb{G} = \tilde{\mathbb{D}}`
  they are :math:`I \otimes \tilde{\mathbb{D}} - \mathbb{A} \otimes \tilde{\mathbb{B}}`. This is
  the descriptor's mass form: a first-order system needs no derivative blocks, because its
  equation gives the derivative (a single field of higher order, :mod:`pihm.assemble.descriptor`,
  carries them). The two forms have the same kernel and differ only by that banded left factor;
  there are no derivative blocks, so there is nothing to rescale;
* the ratio rows of each ``"ratio"`` constraint, :math:`\mathbb{F}/y_a \otimes \mathbb{B}(x^*)`,
  where :math:`y` is the data's moments on the slice (zero on a padding block) and :math:`a` the
  anchor's component (:func:`pihm.operators.constraints.ratio_rows`): they impose
  :math:`z_c(x^*) = (y_c/y_a)\,z_a(x^*)`, pinning the direction of the data and leaving its scale
  free. Divided by :math:`y_a`, the rows are scale-free, as the zero constraint's are, and a
  component whose datum is zero is held to zero exactly as a zero constraint would hold it. They
  span the whole field register, so a circuit builds them from three terms, the identity, a
  projector and a loaded rank-one term;
* :data:`~pihm.assemble.residual.PADDING_WEIGHT` times the identity on each padding block.

The readout decodes the blocks of the problem's own fields (the monomials of degree one), and the
lifted components are auxiliary.

**The lift's level scale**. A Carleman lift carries :math:`w = u/s` with
:math:`s = \max(\|u_0\|, 1)` (:func:`lift_scale`): :math:`F_p` becomes :math:`s^{p-1}F_p`, and
level :math:`k` holds the monomials of :math:`u` over :math:`s^k`. That is the diagonal
similarity :math:`D^{-1}\mathbb{A}D` of the lifted system at every order, so its truncated
solution, and the Carleman error with it, are unchanged, as is :math:`R`. Where
:math:`\|u_0\| > 1` the unscaled levels grow as :math:`\|u_0\|^k`: the state's weight leaves the
fields for the highest level, where no answer-independent initial state reaches it, and the ratio
rows' moments inflate :math:`\alpha`. With the scale no level grows, and a lift of data inside the
unit ball is unchanged. The readout is scaled to the physical datum, so it reads :math:`u`.

**The chain ablation** (``layout="chain"``, the descriptor only): each component also carries its
derivative as a block, tied to it by the recurrence
:math:`-\tilde{\mathbb{D}}z_c + \tilde{\mathbb{B}}z_c' = 0`, and the equations become
:math:`z' - \mathbb{A}z = 0`, as the descriptor carries a single field of higher order
(:mod:`pihm.assemble.descriptor`). The state is the components, then their derivatives, padded to
a power of two, and the derivative blocks are rescaled by the rule for a regular ODE,
:math:`\lambda = 2\rho` with :math:`\rho` the geometric mean of the characteristic roots'
magnitudes, here :math:`\mathbb{A}`'s non-zero eigenvalues (:func:`chain_scale`). Its ratio rows
hold the components only. It is the mass form's classical ablation.

Examples
--------
Rung R7, the oscillator :math:`u' = v,\ v' = -\pi^2 u`, at three qubits:

>>> from pihm.assemble import assemble
>>> from pihm.problems import ladder
>>> residual = assemble(ladder.problem("R7"), 3, regime="descriptor")
>>> residual.monomials, residual.derivatives
(((1, 0), (0, 1)), ((0,), (0,)))
>>> [block.name for block in residual.blocks]
['equation', 'ratio on x = -1']
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm import basis
from pihm.assemble.descriptor import on_block
from pihm.assemble.residual import (
    PADDING_WEIGHT,
    Block,
    FirstOrderLayout,
    Monomial,
    Readout,
    Regime,
    Rescale,
    Residual,
)
from pihm.carleman import carleman_matrix, moments, monomials
from pihm.operators import banded_factors, derivative_matrix, ratio_rows, zero_row
from pihm.problems.polynomial import vector_field
from pihm.problems.reference import reference_solution, regular_value, slice_data
from pihm.problems.spec import Constraint, Problem

__all__ = [
    "assemble_system",
    "chain_scale",
    "components",
    "coupling",
    "lift_scale",
    "ratio_data",
    "system_state",
    "vector_field",
]


def components(problem: Problem, order: int) -> tuple[tuple[int, ...], ...]:
    """Return the monomials a system's blocks carry, before padding: the fields, then the lift's."""
    return monomials(len(problem.fields), order)


def lift_scale(problem: Problem, order: int) -> float:
    r"""Return a Carleman lift's level scale :math:`s = \max(\|u_0\|, 1)`, 1 without a lift.

    :math:`u_0` is the fields' stated values at the ratio constraint, where the lift's data
    are (module documentation).
    """
    if order == 1:
        return 1.0
    norms = [
        float(np.linalg.norm(slice_data(problem, c.axis, c.at)))
        for c in problem.constraints
        if c.kind == "ratio"
    ]
    return max([1.0, *norms])


def coupling(problem: Problem, order: int) -> sp.csr_array:
    r"""Return the system's :math:`\mathbb{A}` on its components (:func:`components`).

    A lift carries :math:`w = u/s` (:func:`lift_scale`), so its :math:`F_p` is
    :math:`s^{p-1}F_p`.
    """
    scale = lift_scale(problem, order)
    field = [F * scale ** (degree - 1) for degree, F in enumerate(vector_field(problem), start=1)]
    return carleman_matrix(field, order)


def chain_scale(problem: Problem, order: int) -> float:
    r"""Return the chain ablation's :math:`\lambda = 2\rho`, the rule for a system.

    :math:`\rho` is the geometric mean of the magnitudes of :math:`\mathbb{A}`'s non-zero
    eigenvalues, the system's characteristic roots; for a single first-order equation
    :math:`u' = -a u` it is :math:`|a|`, as :func:`pihm.problems.scales.block_scales` gives it.
    """
    roots = np.abs(np.linalg.eigvals(coupling(problem, order).toarray()))
    roots = roots[roots > 1e-12 * max(float(np.max(roots)), 1e-300)]
    if not roots.size:
        raise ValueError(f"{problem.key}'s coupling has no non-zero root to scale by")
    return float(2.0 * np.exp(np.mean(np.log(roots))))


def ratio_data(
    problem: Problem, constraint: Constraint, keys: Sequence[tuple[int, ...]], blocks: int
) -> tuple[NDArray[np.float64], int]:
    r"""Return a ratio constraint's data :math:`y` on the field register, and its anchor :math:`a`.

    :math:`y` is the moments of the fields' stated values at the point over the lift's scale
    (:func:`lift_scale`), one per component, and zero on each padding block; :math:`a` is the
    block of the constraint's field.
    """
    order = max(sum(key) for key in keys)
    values = slice_data(problem, constraint.axis, constraint.at) / lift_scale(problem, order)
    data = np.zeros(blocks)
    data[: len(keys)] = moments(values, keys)
    count = len(problem.fields)
    anchor = [f.name for f in problem.fields].index(constraint.field)
    return data, list(keys).index(tuple(int(i == anchor) for i in range(count)))


def _ratio_block(
    constraint: Constraint,
    problem: Problem,
    keys: Sequence[tuple[int, ...]],
    n: int,
    blocks: int,
    held: Sequence[int] = (),
) -> Block:
    r"""Return the rows :math:`\mathbb{F}/y_a \otimes \mathbb{B}(x^*)` of one ratio constraint.

    :math:`\mathbb{F}/y_a = I - |a\rangle\langle a| - |w\rangle\langle a|/y_a` spans the whole
    field register: on a padding block it is the identity, whose rows meet only that block's own
    columns, so the padding still decouples, and its singular values only rise above
    :math:`c_{\rm pad}`. The blocks in ``held`` (the chain ablation's derivatives, whose values
    at the point are not data) keep no row.
    """
    data, a = ratio_data(problem, constraint, keys, blocks)
    ratios = ratio_rows(data, a) / data[a]
    if held:
        kept = np.ones(blocks)
        kept[list(held)] = 0.0
        ratios = sp.diags_array(kept) @ ratios
    ratios = sp.csr_array(ratios)
    name = f"ratio on {constraint.axis} = {constraint.at:.10g}"
    return Block(name, "constraint", sp.csr_array(sp.kron(ratios, zero_row(constraint.at, n))))


def assemble_system(
    problem: Problem,
    n: int,
    regime: Regime,
    rescale: Rescale,
    order: int,
    layout: FirstOrderLayout = "mass",
) -> Residual:
    r"""Assemble a first-order system (module docstring) into its stacked residual.

    Parameters
    ----------
    problem
        A system (:func:`vector_field`), linear or of any polynomial degree.
    n
        Qubits on the axis.
    regime
        ``"standard"``: rows :math:`I \otimes \mathbb{G} - \mathbb{A} \otimes I`;
        ``"descriptor"``: the mass form :math:`I \otimes \tilde{\mathbb{D}} - \mathbb{A} \otimes
        \tilde{\mathbb{B}}`.
    rescale
        The mass form has no derivative blocks, so it only records it; the chain ablation's are
        rescaled by :func:`chain_scale`'s :math:`\lambda` under ``"block"`` (the caller applies
        the weights, as :func:`pihm.assemble.assemble` does).
    order
        The Carleman order :math:`K`, 1 for a linear system.
    layout
        ``"mass"``, or in the descriptor the ``"chain"`` ablation (module documentation).

    Raises
    ------
    NotImplementedError
        For a constraint other than a ratio constraint (the data of a system enter as ratios).
    """
    keys = components(problem, order)
    A = coupling(problem, order)
    count = len(keys)
    chain = layout == "chain"
    carried = 2 * count if chain else count
    blocks = 1 << (carried - 1).bit_length()
    size = 1 << n
    identity = sp.eye_array(size, format="csr")
    values = sp.csr_array(sp.eye_array(count, blocks, format="csr"))
    coupled = sp.csr_array(sp.hstack([A, sp.csr_array((count, blocks - count))]))
    if chain:
        slopes = sp.csr_array(sp.eye_array(count, blocks, k=count, format="csr"))
        B, D = banded_factors(n)
        equation = sp.csr_array(sp.kron(slopes, identity) - sp.kron(coupled, identity))
        recurrence = sp.csr_array(-sp.kron(values, D) + sp.kron(slopes, B))
        rows = [
            Block("equation", "equation", equation),
            Block("recurrence z -> z'", "recurrence", recurrence),
        ]
    else:
        if regime == "standard":
            derivative, mass = derivative_matrix(n), identity
        else:
            mass, derivative = banded_factors(n)
        equation = sp.csr_array(sp.kron(values, derivative) - sp.kron(coupled, mass))
        rows = [Block("equation", "equation", equation)]
    padding = sp.csr_array(PADDING_WEIGHT * sp.eye_array(size, format="csr"))
    rows += [
        Block(f"padding block {i}", "padding", on_block(padding, i, blocks))
        for i in range(carried, blocks)
    ]
    held = range(count, carried)
    for constraint in problem.constraints:
        if constraint.kind != "ratio":
            raise NotImplementedError("a system's data enter as ratio constraints only")
        rows.append(_ratio_block(constraint, problem, keys, n, blocks, held))
    placed: tuple[Monomial, ...] = (*keys, *(keys if chain else ()), *([None] * (blocks - carried)))
    orders = [(0,)] * count + [(1,)] * (carried - count)
    fields = len(problem.fields)
    units = [keys.index(tuple(int(i == j) for i in range(fields))) for j in range(fields)]
    decoder = sp.csr_array(
        sp.vstack([on_block(sp.eye_array(size, format="csr"), u, blocks) for u in units])
    )
    return Residual(
        problem=problem,
        regime=regime,
        nonlinearity="none" if order == 1 else "carleman",
        qubits=(n,),
        lifted=(False,),
        blocks=tuple(rows),
        derivatives=(*orders, *([None] * (blocks - carried))),
        readout=Readout(decoder, (n,), 2),
        regular_value=regular_value(problem),
        mult="lift",
        truncation="none",
        rescale=rescale,
        scales=(chain_scale(problem, order),) if chain and rescale == "block" else (1.0,),
        monomials=placed,
        lift_scale=lift_scale(problem, order),
    )


def system_state(residual: Residual) -> NDArray[Any]:
    r"""Return the reference solution's state in a system's space, unnormalised.

    Each block holds the normalised coefficients of its monomial of the reference fields,
    :math:`\prod_i u_i^{\alpha_i}` (the fields themselves, then in a Carleman lift their exact
    products, not the truncated lift's solution), over the lift's :math:`s^{|\alpha|}`, a chain
    ablation's derivative block that monomial's derivative over its weight, and each padding
    block zero. Its norm squared is the
    ideal scale factor :math:`\eta_e`, whose unit state shares its norm with every block.
    """
    problem = residual.problem
    references = [reference_solution(problem, field.name) for field in problem.fields]
    (n,) = residual.qubits
    blocks = []
    for key, derivative, weight in zip(
        residual.monomials, residual.derivatives, residual.block_weights, strict=True
    ):
        if key is None:
            blocks.append(np.zeros(1 << n))
            continue
        carried = _monomial_coefficients(references, key, n) / residual.lift_scale ** sum(key)
        if derivative != (0,):
            blocks.append(np.asarray(derivative_matrix(n) @ carried) / weight)
        else:
            blocks.append(carried)
    return np.concatenate(blocks)


def _monomial_coefficients(references: Sequence[Any], key: tuple[int, ...], n: int) -> NDArray[Any]:
    """Return the normalised coefficients of the monomial ``key`` of the reference fields."""
    if sum(key) == 1:
        return np.asarray(references[key.index(1)].coefficients(n))

    def product(x: NDArray[Any]) -> NDArray[Any]:
        values = np.ones_like(np.asarray(x, dtype=np.float64))
        for reference, exponent in zip(references, key, strict=True):
            if exponent:
                values = values * np.asarray(reference.field(x)) ** exponent
        return values

    return np.asarray(basis.fit(product, n))
