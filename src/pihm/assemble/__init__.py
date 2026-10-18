r"""Assemble a problem into its stacked residual :math:`R`, with :math:`H = R^\dagger R`.

:func:`assemble` builds the paper's effective Hamiltonian
:math:`\mathbb{H} = \mathcal{T}(\mathbb{A}) + \sum_i \mathcal{T}(\mathbb{B}_i)` as the stacked
residual :math:`R = [\mathbb{A}; \mathbb{B}_1; \dots]`, whose smallest right
singular vector is the ground state (:func:`pihm.classical.reference_state`). It is the one entry
point for the three orthogonal choices, the derivative representation
(``regime``), the treatment of a nonlinearity and the constraint rows, which the problem states:

* :mod:`~pihm.assemble.standard`: the standard form of a single field, exactly as the paper
  builds it, and its doubled space for a quadratic equation (``nonlinearity="doubled"``);
* :mod:`~pihm.assemble.descriptor`: the descriptor form of a single field of higher order, its
  derivatives carried as blocks of the state, block-rescaled;
* :mod:`~pihm.assemble.systems`: a first-order system in either form, several fields or a
  nonlinear field lifted by Carleman (``nonlinearity="carleman"``), and in the
  descriptor form a single first-order equation too, in mass form, with its
  ``layout="chain"`` ablation;
* :mod:`~pihm.assemble.residual`: the :class:`Residual` every builder returns.

Examples
--------
>>> from pihm.problems import ladder
>>> residual = assemble(ladder.problem("R2a"), 3)
>>> residual.matrix().shape, residual.lifted
((9, 8), (False,))
>>> descriptor = assemble(ladder.problem("R2a"), 3, regime="descriptor")
>>> descriptor.derivatives, descriptor.field_qubits, descriptor.matrix().shape
(((0,), (1,), (2,), None), 2, (33, 32))
>>> descriptor.rescale, descriptor.scales, descriptor.block_weights
('block', (4.0,), (1.0, 4.0, 16.0, 1.0))
>>> lifted = assemble(ladder.problem("R8"), 3, carleman=3)
>>> lifted.nonlinearity, lifted.monomials
('carleman', ((1,), (2,), (3,), None))
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import get_args

import numpy as np
import scipy.sparse as sp

from pihm.assemble.descriptor import (
    assemble_descriptor,
    descriptor_layout,
    recurrence_name,
    represent,
)
from pihm.assemble.residual import (
    PADDING_WEIGHT,
    Block,
    BlockKind,
    Derivative,
    FirstOrderLayout,
    Monomial,
    Nonlinearity,
    Readout,
    Regime,
    Rescale,
    Residual,
    Truncation,
)
from pihm.assemble.standard import assemble_doubled, assemble_linear
from pihm.assemble.systems import assemble_system, system_state
from pihm.operators import MultiplicationMode
from pihm.problems.scales import block_scales
from pihm.problems.spec import Problem

__all__ = [
    "PADDING_WEIGHT",
    "Block",
    "BlockKind",
    "Derivative",
    "FirstOrderLayout",
    "Monomial",
    "Nonlinearity",
    "Readout",
    "Regime",
    "Rescale",
    "Residual",
    "Truncation",
    "assemble",
    "first_order",
    "is_system",
    "recurrence_name",
    "represent",
    "system_state",
]


def _treatment(
    problem: Problem, nonlinearity: Nonlinearity | None, carleman: int | None
) -> Nonlinearity:
    """Return the treatment of the nonlinearity: as given, or the one the problem requires."""
    if nonlinearity is not None:
        return nonlinearity
    if carleman is not None:
        return "carleman"
    return "doubled" if problem.degree == 2 else "none"


def first_order(problem: Problem) -> bool:
    """Return whether ``problem`` is a first-order equation or system on one axis."""
    return problem.order == 1 and len(problem.axes) == 1


def is_system(
    problem: Problem,
    regime: Regime = "standard",
    carleman: int | None = None,
    layout: FirstOrderLayout = "mass",
) -> bool:
    """Return whether ``problem`` is assembled as a first-order system.

    That is, by :mod:`pihm.assemble.systems`.

    It is when it has several fields or is lifted by Carleman (``carleman`` given), and in the
    descriptor form also a single first-order equation on one axis carried in its mass form: its
    equation gives its derivative, so it needs no derivative block. Its chain
    ablation is the descriptor's chain for a single field (:mod:`pihm.assemble.descriptor`).
    """
    return (
        len(problem.fields) > 1
        or carleman is not None
        or (regime == "descriptor" and first_order(problem) and layout == "mass")
    )


def _check_options(
    problem: Problem,
    regime: Regime,
    mult: MultiplicationMode,
    truncation: Truncation,
    nonlinearity: Nonlinearity,
    rescale: Rescale,
    carleman: int | None,
    layout: FirstOrderLayout,
) -> None:
    """Refuse combinations that are not implemented, or not consistent, with the reason."""
    for value, allowed in (
        (regime, Regime),
        (mult, MultiplicationMode),
        (truncation, Truncation),
        (nonlinearity, Nonlinearity),
        (rescale, Rescale),
        (layout, FirstOrderLayout),
    ):
        if value not in get_args(allowed):
            raise ValueError(f"{value!r} is not one of {get_args(allowed)}")
    if (nonlinearity == "carleman") != (carleman is not None):
        raise ValueError("a Carleman lift needs its order, and only a Carleman lift takes one")
    if carleman is not None and carleman < 1:
        raise ValueError(f"the Carleman order must be at least 1, got {carleman}")
    if nonlinearity != "none" and problem.is_linear:
        raise ValueError(f"{problem.key} is linear: use nonlinearity='none'")
    if nonlinearity != "carleman" and problem.degree > 2:
        raise NotImplementedError("the doubled space is implemented for quadratic equations")
    if nonlinearity == "none" and not problem.is_linear:
        kind = "quadratic" if problem.degree == 2 else "nonlinear"
        raise ValueError(f"{problem.key} is {kind}: use nonlinearity='doubled' or 'carleman'")
    if layout == "chain" and (regime != "descriptor" or not first_order(problem)):
        raise ValueError(
            "the chain layout is the descriptor's ablation of a first-order equation or system"
        )
    system = is_system(problem, regime, carleman, layout)
    if system and (mult != "lift" or truncation != "none"):
        raise ValueError("a system takes the default multiplication and no truncation")
    if not system and layout == "mass" and any(c.kind == "ratio" for c in problem.constraints):
        raise ValueError(
            f"{problem.key}'s ratio rows pin a system's data: assemble it with a Carleman lift"
        )
    if regime == "descriptor" and nonlinearity == "doubled":
        raise NotImplementedError(
            "the descriptor form of the doubled space (R6) is not implemented"
        )
    if regime == "descriptor" and truncation != "none":
        raise ValueError("the tau truncation is an ablation of the standard form")
    if regime == "standard" and rescale != "none":
        raise ValueError("the standard form's state is one block, so it has no block rescaling")


def assemble(
    problem: Problem,
    n: int | Sequence[int],
    *,
    regime: Regime = "standard",
    mult: MultiplicationMode = "lift",
    truncation: Truncation = "none",
    nonlinearity: Nonlinearity | None = None,
    rescale: Rescale | None = None,
    carleman: int | None = None,
    layout: FirstOrderLayout = "mass",
) -> Residual:
    r"""Assemble ``problem`` at resolution ``n`` into its stacked residual.

    Parameters
    ----------
    problem
        A problem: one field, or a first-order system of several (:mod:`pihm.assemble.systems`).
    n
        Qubits per axis of the field: one integer for every axis, or one per axis.
    regime
        ``"standard"``, the paper's, where derivatives are :math:`\mathbb{G}^j`; or
        ``"descriptor"``, where they are blocks of the state tied by banded recurrences, or for a
        system the equations' banded mass form.
    mult
        ``"lift"``: lift the axes that need it, as the paper does; ``"square"``: never lift.
    truncation
        ``"none"`` (the paper's residual) or ``"tau"`` (drop the equation block's top rows; the
        standard form only).
    nonlinearity
        ``"none"`` for a linear problem; for a nonlinear one the paper's ``"doubled"`` space
        (quadratic) or the ``"carleman"`` lift. By default ``"carleman"`` when ``carleman`` is
        given, and otherwise the one the problem's degree requires.
    rescale
        The descriptor's block rescaling: ``"block"`` (its default) or ``"none"``,
        the unscaled ablation. The standard form takes ``"none"``, its default.
    carleman
        The Carleman lift's order :math:`K` (``nonlinearity="carleman"`` only).
    layout
        How the descriptor carries a first-order equation or system: its ``"mass"`` form, or the
        ``"chain"`` ablation, each component with its derivative block,
        block-rescaled under ``rescale="block"``.

    Returns
    -------
    Residual
        The row blocks, the state's blocks, the readout and the options used.

    Raises
    ------
    NotImplementedError
        For a nonlinearity or layout that is not implemented: a cubic term outside a Carleman
        lift, the descriptor form of a doubled space, a mixed derivative in the descriptor form,
        or a system outside :func:`~pihm.assemble.systems.vector_field`'s class.
    ValueError
        For an inconsistent choice of options.
    """
    treatment = _treatment(problem, nonlinearity, carleman)
    scaling: Rescale = rescale or ("block" if regime == "descriptor" else "none")
    _check_options(problem, regime, mult, truncation, treatment, scaling, carleman, layout)
    axes = problem.fields[0].axes
    ns = (int(n),) * len(axes) if isinstance(n, (int, np.integer)) else tuple(int(k) for k in n)
    if len(ns) != len(axes) or any(k < 1 for k in ns):
        raise ValueError(f"need one qubit count >= 1 per axis {axes}, got {n!r}")
    if is_system(problem, regime, carleman, layout):
        if len(ns) != 1:
            raise NotImplementedError("systems are implemented on one axis")
        system = assemble_system(problem, ns[0], regime, scaling, carleman or 1, layout)
        return _rescaled(system) if layout == "chain" and scaling == "block" else system
    derivatives: tuple[Derivative, ...]
    if regime == "descriptor":
        build, derivatives = assemble_descriptor, descriptor_layout(problem)
    else:
        build = assemble_doubled if treatment == "doubled" else assemble_linear
        derivatives = ((0,) * len(ns),)
    blocks, lifted, readout, value = build(problem, ns, mult, truncation)
    residual = Residual(
        problem=problem,
        regime=regime,
        nonlinearity=treatment,
        qubits=ns,
        lifted=lifted,
        blocks=blocks,
        derivatives=derivatives,
        readout=readout,
        regular_value=value,
        mult=mult,
        truncation=truncation,
        rescale=scaling,
        scales=block_scales(problem, ns) if scaling == "block" else (1.0,) * len(ns),
    )
    return _rescaled(residual) if scaling == "block" else residual


def _rescaled(residual: Residual) -> Residual:
    """Return ``residual`` with every row block's columns scaled by the block weights."""
    size = residual.dimension // len(residual.derivatives)
    S = sp.diags_array(np.repeat(residual.block_weights, size), format="csr")
    blocks = tuple(
        Block(block.name, block.kind, sp.csr_array(block.matrix @ S)) for block in residual.blocks
    )
    return dataclasses.replace(residual, blocks=blocks)
