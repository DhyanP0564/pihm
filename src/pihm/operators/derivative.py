r"""The Chebyshev derivative :math:`\mathbb{G}`, its banded factors and its exact factorisation.

In the normalised basis, differentiation :math:`f \mapsto f'` acts on coefficients as

.. math::

   \mathbb{G}_{kp} = \frac{2p}{\sqrt{c_k}} \quad (p > k,\ p - k \text{ odd}), \qquad
   c_0 = 2,\ c_{k \ge 1} = 1,

which is strictly upper triangular, nilpotent and dense in the odd diagonals, with
:math:`\|\mathbb{G}\|_2 = \Theta(N^2)`. The paper prints this matrix as
:math:`\mathbb{G}_n^T` (supplementary §A), because it applies :math:`\mathbb{G}_n^T` to the
state. The two factorisations here are the structural facts the rest of this library uses:

* :func:`banded_factors`: :math:`\mathbb{G} = \tilde{\mathbb{B}}^{-1}\tilde{\mathbb{D}}` with
  both factors banded (the descriptor regime keeps only the factors, carrying derivatives as
  unknowns or, for a first-order equation, multiplying it through by :math:`\tilde{\mathbb{B}}`)
  :cite:`olver2013ultraspherical`;
* :func:`derivative_factorisation`: :math:`\mathbb{G} = R_0\,(2U_{\rm odd})\,\Lambda`, the
  exact structure behind the standard-form regime's best block encoding.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pihm.basis import chebyshev_scale

__all__ = ["banded_factors", "derivative_factorisation", "derivative_matrix"]


def _size(n: int) -> int:
    """Return :math:`N = 2^n` after validating ``n`` (by building the scale vector)."""
    return int(chebyshev_scale(n).size)


def derivative_matrix(n: int) -> sp.csr_array:
    r"""Return the derivative matrix :math:`\mathbb{G}` in the normalised basis.

    Parameters
    ----------
    n
        Qubits in the register.

    Returns
    -------
    scipy.sparse.csr_array
        The ``(N, N)`` matrix with :math:`\mathbb{G}_{kp} = 2p/\sqrt{c_k}` for :math:`p > k`
        and :math:`p - k` odd. If ``b`` holds the normalised coefficients of :math:`f`, then
        ``G @ b`` holds those of :math:`f'`.

    Examples
    --------
    The paper's printed :math:`\mathbb{G}_2^T`:

    >>> derivative_matrix(2).toarray()
    array([[0.        , 1.41421356, 0.        , 4.24264069],
           [0.        , 0.        , 4.        , 0.        ],
           [0.        , 0.        , 0.        , 6.        ],
           [0.        , 0.        , 0.        , 0.        ]])
    """
    size = _size(n)
    k, p = np.triu_indices(size, 1)
    odd = (p - k) % 2 == 1
    k, p = k[odd], p[odd]
    values = 2.0 * p / np.where(k == 0, np.sqrt(2.0), 1.0)
    return sp.csr_array((values, (k, p)), shape=(size, size))


def banded_factors(n: int) -> tuple[sp.csr_array, sp.csr_array]:
    r"""Return the banded factors :math:`\tilde{\mathbb{B}}, \tilde{\mathbb{D}}` of the derivative.

    The standard Chebyshev coefficients :math:`a` of :math:`f` and :math:`p` of :math:`f'` obey

    .. math::

       c_r\,p_r - p_{r+2} = 2(r+1)\,a_{r+1}, \qquad r = 0, \dots, N-1,

    that is :math:`\tilde{\mathbb{B}}p = \tilde{\mathbb{D}}a` with :math:`p_{N} = p_{N+1} = 0`.
    Conjugating by :math:`S = \operatorname{diag}(\sqrt{c_k})` moves both factors into the
    normalised basis. :math:`\tilde{\mathbb{B}}` is upper triangular with a non-zero diagonal,
    hence invertible, and :math:`\tilde{\mathbb{B}}\,\mathbb{G} = \tilde{\mathbb{D}}` exactly.

    Parameters
    ----------
    n
        Qubits in the register.

    Returns
    -------
    B_tilde, D_tilde
        Sparse ``(N, N)`` factors. :math:`\tilde{\mathbb{B}}` occupies the diagonal
        (:math:`2, 1, 1, \dots`) and the second superdiagonal (:math:`-\sqrt2, -1, -1, \dots`);
        :math:`\tilde{\mathbb{D}}` occupies the first superdiagonal
        (:math:`2\sqrt2, 4, 6, \dots`).

    See Also
    --------
    derivative_matrix : The dense product :math:`\tilde{\mathbb{B}}^{-1}\tilde{\mathbb{D}}`.
    """
    size = _size(n)
    c = np.ones(size)
    c[0] = 2.0
    root = np.sqrt(c)
    r = np.arange(size)
    upper = np.arange(size - 2)
    B_raw = sp.csr_array(
        (
            np.concatenate([c, -np.ones(upper.size)]),
            (np.concatenate([r, upper]), np.concatenate([r, upper + 2])),
        ),
        shape=(size, size),
    )
    first = np.arange(size - 1)
    D_raw = sp.csr_array((2.0 * (first + 1), (first, first + 1)), shape=(size, size))
    S = sp.diags_array(root)
    S_inv = sp.diags_array(1.0 / root)
    return sp.csr_array(S @ B_raw @ S_inv), sp.csr_array(S @ D_raw @ S_inv)


def derivative_factorisation(n: int) -> tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
    r"""Return the factors of :math:`\mathbb{G} = R_0\,(2U_{\rm odd})\,\Lambda`.

    The three factors are each easy to block-encode exactly:

    * :math:`R_0 = I - (1 - 1/\sqrt2)\,|0\rangle\langle 0|`, a two-term LCU of :math:`I` and
      the reflection :math:`I - 2|0\rangle\langle0|` with :math:`\alpha = 1`;
    * :math:`(U_{\rm odd})_{kp} = [\,p > k,\ p - k \text{ odd}\,]`, the sum of the :math:`N/2`
      truncated shifts by odd offsets (an arithmetic SELECT with :math:`\alpha = N/2`);
    * :math:`\Lambda = \operatorname{diag}(p) = \tfrac{N-1}{2} I - \sum_j 2^{j-1} Z_j`, a
      Pauli-Z LCU with :math:`\alpha = N - 1`.

    So :math:`\alpha_{\mathbb{G}} = N(N-1)`, within a small constant of
    :math:`\|\mathbb{G}\|_2`.

    Parameters
    ----------
    n
        Qubits in the register.

    Returns
    -------
    R0, U_odd, Lambda
        Sparse ``(N, N)`` factors.
    """
    size = _size(n)
    R0 = sp.diags_array(np.where(np.arange(size) == 0, 1.0 / np.sqrt(2.0), 1.0), format="csr")
    k, p = np.triu_indices(size, 1)
    odd = (p - k) % 2 == 1
    U_odd = sp.csr_array((np.ones(int(odd.sum())), (k[odd], p[odd])), shape=(size, size))
    Lambda = sp.diags_array(np.arange(size, dtype=np.float64), format="csr")
    return R0, U_odd, Lambda
