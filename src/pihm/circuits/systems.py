r"""Compile a first-order system into circuit block encodings, in either form.

A system (:mod:`pihm.assemble.systems`) has a field register of :math:`n_f` qubits, one block per
component, above its axis's register. Its stacked residual is the system rows and each ratio
constraint's rows, built from the problem at any :math:`n`:

* **the system rows**, the equation and the padding:

  - in the descriptor form, the mass form
    :math:`\Pi_C\otimes\tilde{\mathbb{D}} - \mathbb{A}\otimes\tilde{\mathbb{B}} +
    c_{\rm pad}\Pi_{\rm pad}\otimes I`, whose axis factors are all bands: its term IR
    (:func:`pihm.ir.system_ir`) compiled by :func:`~pihm.circuits.lcu.compile_system`;
  - in the standard form, :math:`\Pi_C\otimes\mathbb{G} + (c_{\rm pad}\Pi_{\rm pad} -
    \mathbb{A})\otimes I`, with :math:`\mathbb{G}` the standard form's own encoder
    (:data:`~pihm.circuits.standard.DERIVATIVE_ENCODERS`): two terms, each field-register matrix
    compiled alone (:func:`field_encoding`) and placed beside the axis's factor, summed by
    :func:`~pihm.circuits.compose.lcu_sum`. Without padding :math:`\Pi_C = I`, and the first
    term is :math:`\mathbb{G}` alone;

  :math:`\Pi_C` is the components' projector on the field register and :math:`\Pi_{\rm pad}` the
  padding's. The two forms pay the same for the coupling :math:`\mathbb{A}`, a sum of matchings
  in both (:func:`pihm.ir.matching_terms`), and :math:`\Pi_C` is in Pauli words
  (:func:`pihm.ir.field_terms`).
* **each ratio constraint's rows**,
  :math:`(I - |a\rangle\langle a| - \|w\|\,|\hat w\rangle\langle a|)\otimes\mathbb{B}(x^*)`
  with :math:`w` the data over its anchor :math:`y_a` off the anchor
  (:func:`pihm.assemble.systems.ratio_data`): three terms, the identity, a projector
  (:func:`~pihm.circuits.lcu.field_placement`) and a rank-one term whose :math:`\hat w` a rotation
  tree loads (:func:`~pihm.circuits.compose.real_state_preparation`), each beside the zero row
  (:func:`~pihm.circuits.standard.zero_encoding`). The third is absent when every other datum is
  zero, as R7's is.

The rows fill the system register, field register first, as the residual lays its state out, so
the stacked reflection encodes :math:`2H/\alpha_R^2 - I` for the assembled :math:`H`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pihm.assemble import PADDING_WEIGHT, Regime, Rescale, assemble
from pihm.assemble.systems import components, coupling, ratio_data
from pihm.circuits.compose import (
    lcu_sum,
    product,
    real_state_preparation,
    stack_rows,
    stacked_reflection,
)
from pihm.circuits.lcu import Layout, compile_system, field_placement, place
from pihm.circuits.primitives import BlockEncoding
from pihm.circuits.standard import DERIVATIVE_ENCODERS, StandardEncoder, zero_encoding
from pihm.ir import SystemIR, field_terms, matching_terms, system_ir
from pihm.problems.spec import Constraint, Problem

__all__ = ["encode_system", "field_encoding", "ratio_encoding", "system_blocks"]


def field_encoding(
    F: NDArray[np.float64], field_qubits: int, label: str = "", *, matchings: bool = False
) -> BlockEncoding:
    r"""Return a block encoding of the real matrix :math:`F` on the field register alone.

    :math:`F`'s Pauli words (:func:`pihm.ir.field_terms`), or its matchings
    (:func:`pihm.ir.matching_terms`), compiled as one linear combination
    (:func:`~pihm.circuits.lcu.compile_system`); :math:`\alpha` is the sum of their coefficients'
    magnitudes.
    """
    terms = matching_terms(F) if matchings else field_terms(F)
    return compile_system(SystemIR(tuple(terms), field_qubits, (), ()), label=label)


def _layout(problem: Problem, n: int, order: int) -> tuple[Layout, int, int]:
    """Return a system's layout, its number of components and its number of blocks."""
    count = len(components(problem, order))
    blocks = 1 << (count - 1).bit_length()
    return Layout((blocks - 1).bit_length(), (n,), (n,)), count, blocks


def _standard_rows(problem: Problem, n: int, order: int, encoder: StandardEncoder) -> BlockEncoding:
    r"""Return :math:`\Pi_C\otimes\mathbb{G} + (c_{\rm pad}\Pi_{\rm pad} - \mathbb{A})\otimes I`."""
    layout, count, blocks = _layout(problem, n, order)
    G = DERIVATIVE_ENCODERS[encoder](n)
    kept = np.diag((np.arange(blocks) < count).astype(np.float64))
    if count == blocks:
        derivative = place([(G, layout.data(0))], layout, label="I (x) G")
    else:
        projector = field_encoding(kept, layout.field_qubits, label="Pi_C")
        derivative = place(
            [(projector, layout.field()), (G, layout.data(0))], layout, label="Pi_C (x) G"
        )
    rest = PADDING_WEIGHT * (np.eye(blocks) - kept)
    rest[:count, :count] -= coupling(problem, order).toarray()
    field = field_encoding(rest, layout.field_qubits, label="c_pad Pi_pad - A", matchings=True)
    other = place([(field, layout.field())], layout, label="(c_pad Pi_pad - A) (x) I")
    return lcu_sum([derivative, other], [1.0, 1.0], label="A_sys")


def ratio_encoding(
    problem: Problem, constraint: Constraint, n: int, order: int, label: str = ""
) -> BlockEncoding:
    r"""Return a ratio constraint's rows (module documentation).

    :math:`(I - |a\rangle\langle a| - |w\rangle\langle a|)\otimes\mathbb{B}(x^*)`, three terms.
    """
    layout, _, blocks = _layout(problem, n, order)
    data, a = ratio_data(problem, constraint, components(problem, order), blocks)
    w = data / data[a]
    w[a] = 0.0
    row = zero_encoding(constraint.at, n)
    axis = layout.data(0)
    field = layout.field()
    terms = [
        place([(row, axis)], layout, label="I (x) B"),
        place([(field_placement(a, a, layout.field_qubits), field), (row, axis)], layout),
    ]
    weights: list[complex] = [1.0, -1.0]
    norm = float(np.linalg.norm(w))
    if norm > 0.0:
        size = 1 << layout.field_qubits
        loader = BlockEncoding(
            real_state_preparation(w / norm, "S_w"), 1.0, layout.field_qubits, 0, (size, size)
        )
        unit = product(loader, field_placement(0, a, layout.field_qubits), label="|w><a|")
        terms.append(place([(unit, field), (row, axis)], layout, label="|w><a| (x) B"))
        weights.append(-norm)
    return lcu_sum(terms, weights, label=label or "ratio")


def system_blocks(
    problem: Problem,
    n: int,
    *,
    regime: Regime,
    rescale: Rescale | None = None,
    carleman: int | None = None,
    encoder: StandardEncoder = "structured",
) -> list[BlockEncoding]:
    r"""Return a system's row blocks: the system rows, then each ratio constraint's.

    Parameters
    ----------
    problem
        A system (:func:`pihm.assemble.is_system`).
    n
        Qubits on its axis.
    regime
        The form.
    rescale
        As :func:`pihm.assemble.assemble` takes it; a system has nothing to rescale.
    carleman
        The Carleman lift's order, if the problem is lifted.
    encoder
        The standard form's encoder of :math:`\\mathbb{G}`.

    Raises
    ------
    NotImplementedError
        For a constraint other than a ratio constraint.
    """
    order = carleman or 1
    if regime == "descriptor":
        residual = assemble(problem, n, regime=regime, rescale=rescale, carleman=carleman)
        rows = compile_system(system_ir(residual), label="A_sys")
    else:
        rows = _standard_rows(problem, n, order, encoder)
    blocks = [rows]
    for constraint in problem.constraints:
        if constraint.kind != "ratio":
            raise NotImplementedError("a system's data enter as ratio constraints only")
        blocks.append(ratio_encoding(problem, constraint, n, order))
    return blocks


def encode_system(
    problem: Problem,
    n: int | Sequence[int],
    *,
    regime: Regime,
    rescale: Rescale | None = None,
    carleman: int | None = None,
    encoder: StandardEncoder = "structured",
) -> tuple[BlockEncoding, BlockEncoding]:
    r"""Return a system's stacked residual :math:`U_R` and its reflection :math:`V`.

    As :func:`pihm.circuits.assemble.encode_linear` returns them, for the rows of
    :func:`system_blocks`.
    """
    (size,) = (n,) if isinstance(n, int) else tuple(n)
    blocks = system_blocks(
        problem, size, regime=regime, rescale=rescale, carleman=carleman, encoder=encoder
    )
    stacked = stack_rows(blocks)
    return stacked, stacked_reflection(stacked)
