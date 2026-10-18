r"""The assembled residual: its row blocks, its readout, and the options that produced it.

Every builder of :mod:`pihm.assemble` returns a :class:`Residual`: the row blocks of the stacked
:math:`R`, whose Hamiltonian is :math:`H = R^\dagger R`, the layout of the
state they act on, and the :class:`Readout` that turns a state into the model function.

**The state's blocks.** The state is a list of equal blocks, the most significant index (the field
register of :math:`n_f` qubits), each holding coefficients on the axis registers:

* one block, the field itself, in the standard form of a single field;
* the field and its derivatives, padded to a power of two, in the descriptor form
  (:attr:`Residual.derivatives`);
* one block per component in a first-order system: each field, or in a Carleman lift each
  monomial of the fields up to the lift's order, padded to a power of two
  (:attr:`Residual.monomials`).

A padding block decouples from the rest exactly, so :attr:`Residual.physical_columns` hold the
whole of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm.operators import MultiplicationMode
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
]

#: The derivative representation.
Regime: TypeAlias = Literal["standard", "descriptor"]

#: The treatment of a nonlinear equation, ``"none"`` for a linear problem, the paper's
#: ``"doubled"`` space, or the ``"carleman"`` lift to a linear first-order system.
Nonlinearity: TypeAlias = Literal["none", "doubled", "carleman"]

#: Whether the equation block keeps every row (the paper) or drops the top ``order`` rows.
Truncation: TypeAlias = Literal["none", "tau"]

#: Whether the descriptor's derivative blocks are rescaled, ``"block"``, or not, the
#: ``"none"`` of the unscaled ablation. The standard form's state is one block: ``"none"`` only.
Rescale: TypeAlias = Literal["none", "block"]

#: How the descriptor carries a first-order equation or system. Its ``"mass"`` form has no
#: derivative block; the ``"chain"`` ablation carries each component with its
#: derivative as a block, as the descriptor carries a field of higher order.
FirstOrderLayout: TypeAlias = Literal["mass", "chain"]

#: The role of a row block. In the descriptor form a block may also be a recurrence between two
#: derivative blocks, or the rows that pin a padding block to zero.
BlockKind: TypeAlias = Literal["equation", "constraint", "recurrence", "padding"]

#: The derivative a block of the state carries, one order per axis (all zero for the field), or
#: ``None`` for a padding block.
Derivative: TypeAlias = tuple[int, ...] | None

#: The monomial of a system's fields that a block carries, as exponents in the problem's field
#: order (:math:`u` is ``(1, 0)`` and :math:`uv` is ``(1, 1)`` for fields :math:`u, v`), or
#: ``None`` for a padding block.
Monomial: TypeAlias = tuple[int, ...] | None

#: The weight :math:`c_{\rm pad}` of the rows that pin each padding block to zero.
#: A padding block decouples from the rest of the state exactly, so it
#: adds the singular value :math:`c_{\rm pad}` once per coefficient and nothing else. 1 lies above
#: the first excited singular value of every padded panel of the ladder (at most 0.74, campaign
#: ``p5_build``), so the padding never sets the gap.
PADDING_WEIGHT: Final[float] = 1.0


@dataclass(frozen=True)
class Block:
    """One row block of the stacked residual.

    Attributes
    ----------
    name
        A readable label, such as ``"equation"`` or ``"zero f^(1) on x = -0.75"``.
    kind
        The block's role, one of ``BlockKind``.
    matrix
        The block, with one column per state coefficient.
    """

    name: str
    kind: BlockKind
    matrix: sp.csr_array


@dataclass(frozen=True)
class Readout:
    r"""How a state becomes the model functions :math:`f_q`, and how the scale enters.

    Attributes
    ----------
    decoder
        Maps the state to the normalised coefficients of :math:`f_q` on :attr:`qubits`: of the
        problem's fields one after another, in the problem's order.
    qubits
        The register size of the decoder's output, per axis.
    power
        :math:`p` in :math:`f_Q = \eta^{1/p} f_q`: 2 for the linear model (Eq. 2,
        :math:`\sqrt\eta`) and 1 for the doubled space (Eq. 30, :math:`\eta`).
    """

    decoder: sp.csr_array
    qubits: tuple[int, ...]
    power: int


@dataclass(frozen=True)
class Residual:
    r"""The stacked residual of a problem at a resolution, in one regime.

    Attributes
    ----------
    problem
        The problem.
    regime
        The derivative representation.
    nonlinearity
        The treatment of the nonlinearity.
    qubits
        Qubits per axis of the field (:math:`n`).
    lifted
        Per axis, whether the equation block lives on :math:`n + 1` qubits.
    blocks
        The row blocks, equation first.
    derivatives
        The blocks of the state, in order, by the derivative each carries: one block, the field
        itself, in the standard form; the field and its derivatives, padded to a power of two, in
        the descriptor form; a system's components, each carrying itself.
    readout
        The decoder of the model functions.
    regular_value
        The value :math:`u(x_s)` used in every :math:`\mathbb{D}^{(0)}(x_s)` and datum anchor.
    mult
        The multiplication mode used.
    truncation
        The row truncation used.
    rescale
        The block rescaling used.
    scales
        Its :math:`\lambda_a`, one per axis (all 1 when the residual is not rescaled, and for a
        system, whose blocks carry no derivatives).
    monomials
        For a first-order system (several fields, or a Carleman lift), the monomial each block
        carries, as ``Monomial`` exponents; empty for a single field's residual, whose every block
        carries the field or a derivative of it.
    lift_scale
        A Carleman lift's level scale :math:`s`: its state carries :math:`w = u/s`, level
        :math:`k` the monomials of :math:`u` over :math:`s^k`
        (:func:`pihm.assemble.systems.lift_scale`); 1 without a lift.
    """

    problem: Problem
    regime: Regime
    nonlinearity: Nonlinearity
    qubits: tuple[int, ...]
    lifted: tuple[bool, ...]
    blocks: tuple[Block, ...]
    derivatives: tuple[Derivative, ...]
    readout: Readout
    regular_value: float
    mult: MultiplicationMode
    truncation: Truncation
    rescale: Rescale
    scales: tuple[float, ...]
    monomials: tuple[Monomial, ...] = ()
    lift_scale: float = 1.0

    @property
    def dimension(self) -> int:
        """The number of state coefficients (the columns of :math:`R`)."""
        return int(self.blocks[0].matrix.shape[1])

    @property
    def field_qubits(self) -> int:
        """The qubits :math:`n_f` of the register that indexes the state's blocks."""
        return (len(self.derivatives) - 1).bit_length()

    @property
    def is_system(self) -> bool:
        """Whether the state is a first-order system's components (:attr:`monomials`)."""
        return bool(self.monomials)

    @property
    def carleman_order(self) -> int:
        """The Carleman lift's order :math:`K`, the largest degree carried (1 without a lift)."""
        return max((sum(m) for m in self.monomials if m is not None), default=1)

    @property
    def block_weights(self) -> tuple[float, ...]:
        r"""Each block's column factor :math:`\prod_a\lambda_a^{m_a}` (1 for a padding block)."""
        return tuple(
            1.0
            if derivative is None
            else float(np.prod([lam**m for lam, m in zip(self.scales, derivative, strict=True)]))
            for derivative in self.derivatives
        )

    @property
    def physical_columns(self) -> NDArray[np.intp]:
        """The state's coefficients outside the padding blocks, which decouple from them exactly."""
        size = self.dimension // len(self.derivatives)
        return np.concatenate(
            [
                np.arange(index * size, (index + 1) * size)
                for index, derivative in enumerate(self.derivatives)
                if derivative is not None
            ]
        )

    @property
    def field_blocks(self) -> tuple[int, ...]:
        """The blocks that carry the problem's fields themselves, in the problem's field order.

        The field block (block 0) of a single field's residual; a system's blocks of degree one,
        which are its fields also in a Carleman lift.
        """
        if not self.monomials:
            return (self.derivatives.index((0,) * len(self.qubits)),)
        count = len(self.problem.fields)
        units = [tuple(int(i == j) for i in range(count)) for j in range(count)]
        return tuple(self.monomials.index(unit) for unit in units)

    def matrix(self, *, compact: bool = True) -> sp.csr_array:
        r"""Return the stacked residual :math:`R`.

        Parameters
        ----------
        compact
            Drop rows that are identically zero. They contribute nothing to
            :math:`R^\dagger R`, so the spectrum and the ground state are unchanged, and a
            rank-one constraint on a slice has many of them.
        """
        stacked = sp.csr_array(sp.vstack([block.matrix for block in self.blocks], format="csr"))
        if not compact:
            return stacked
        stacked.eliminate_zeros()
        keep = np.flatnonzero(np.diff(stacked.indptr))
        return sp.csr_array(stacked[keep])

    def hamiltonian(self) -> sp.csr_array:
        r"""Return :math:`H = R^\dagger R`, for inspection only (never for the reference state)."""
        R = self.matrix()
        return sp.csr_array(R.conj().T @ R)
