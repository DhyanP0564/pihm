r"""Source injection as circuits: the block :math:`\mathbb{M}_r\mathbb{D}^{(0)}(x_s)`.

The circuit counterpart of :func:`pihm.assemble.assemble`'s source term. For
:math:`\sum_ja_jf^{(j)}=r(x)` the paper's equation block carries :math:`-\mathbb{M}_r
\mathbb{D}^{(0)}(x_s)`: :math:`\mathbb{D}^{(0)}(x_s)=\mathbb{B}(x_s)/f(x_s)` maps the state to the
constant function :math:`f_q(x_s)/f(x_s)` (:func:`~pihm.circuits.standard.collapse_encoding`), and
the lift :math:`\mathbb{M}_r` multiplies it by :math:`r`, so the block is the rank-one
:math:`|\hat r\rangle\langle\tau(x_s)|/f(x_s)`, :math:`\hat r` the coefficients of :math:`r` on the
lifted register. It is built two ways, one per representation of :math:`r`:

* **A polynomial** (a Maclaurin truncation, as the paper prints, or a polynomial multiplier on a
  term without factors): the paper's :math:`\sum_pc_p\mathbb{M}_{x^p}\mathbb{D}^{(0)}(x_s)`
  (Eq. 14), :func:`~pihm.circuits.multiplication.polynomial_encoding` after
  :func:`~pihm.circuits.multiplication.embed`. Exact while :math:`\deg r<2N`, since the lift acts
  on a constant.
* **A Chebyshev series** (the corrected variants' Chebyshev-accurate :math:`\mathbb{M}_r`): the
  collapse leaves only :math:`|0\rangle` on the lifted register, so :math:`\mathbb{M}_r
  \mathbb{D}^{(0)} = (\|\hat r\|/\sqrt N)\,S_r\,[\mathbb{D}^{(0)};\,0]` for any unitary :math:`S_r`
  with :math:`S_r|0\rangle=\hat r/\|\hat r\|`, here
  :func:`~pihm.circuits.compose.real_state_preparation` of the same :math:`\hat r` the classical
  assembly uses (:func:`pihm.problems.functions.source_coefficients`).
"""

from __future__ import annotations

import numpy as np

from pihm.circuits.compose import pad_rows, product, real_state_preparation
from pihm.circuits.multiplication import embed, polynomial_encoding
from pihm.circuits.primitives import BlockEncoding
from pihm.circuits.standard import collapse_encoding
from pihm.problems.expressions import compile_expression
from pihm.problems.functions import maclaurin_coefficients, source_coefficients
from pihm.problems.reference import regular_value
from pihm.problems.spec import Problem, Term

__all__ = ["source_encoding"]


def source_encoding(term: Term, problem: Problem, n: int) -> BlockEncoding:
    r"""Return :math:`\mathbb{M}_r\mathbb{D}^{(0)}(x_s)` for a source term, without its coefficient.

    Parameters
    ----------
    term
        A term without factors: a source, or a polynomial multiplier alone.
    problem
        The one-axis problem it belongs to, for its regular point :math:`x_s` and value.
    n
        Qubits in the Chebyshev register.

    Returns
    -------
    BlockEncoding
        Shape ``(2N, N)`` on :math:`n+1` system qubits.

    Raises
    ------
    ValueError
        If ``term`` has factors, or both a source and a multiplier.
    NotImplementedError
        For a complex source.
    """
    if term.factors:
        raise ValueError("a source term has no factors")
    if term.source is not None and term.multiplier is not None:
        raise ValueError("a source term carries a source or a multiplier, not both")
    (axis,) = problem.fields[0].axes
    D0 = collapse_encoding(problem.regular.at[0], regular_value(problem), n)
    if term.source is None:
        assert term.multiplier is not None  # a term has factors, a source or a multiplier
        return product(polynomial_encoding(term.multiplier.coefficients, n + 1), embed(D0))
    if term.source.representation == "maclaurin":
        assert term.source.degree is not None  # guaranteed by Source's validation
        compiled = compile_expression(term.source.expression, [axis])
        series = maclaurin_coefficients(compiled, term.source.degree)
        return product(polynomial_encoding(tuple(float(c) for c in series), n + 1), embed(D0))
    coefficients = source_coefficients(term.source, [axis], [n + 1])
    if np.iscomplexobj(coefficients) and np.any(coefficients.imag != 0):
        raise NotImplementedError("source_encoding loads a real source only")
    r_hat = np.real(coefficients)
    size = r_hat.size
    loader = BlockEncoding(real_state_preparation(r_hat, "S_r"), 1.0, n + 1, 0, (size, size), "S_r")
    block = product(loader, pad_rows(D0))
    scale = float(np.linalg.norm(r_hat)) / np.sqrt(size // 2)
    return BlockEncoding(
        block.circuit,
        block.alpha * scale,
        block.n_sys_qubits,
        block.n_anc_qubits,
        block.shape,
        "M_r·D0",
    )
