r"""The iterative certified reference, past the dense SVD's ceiling.

Beyond :data:`DENSE_COLUMNS` physical columns a dense SVD takes minutes and gigabytes, while the
descriptor's :math:`R` stays sparse. :func:`iterative_reference_state` computes what the dense
reference gives the pipeline, the smallest singular triplets, the kernel's gap ratio and
:math:`\sigma_{\max}`, from sparse operations alone:

* **The smallest triplets.** Shift-invert Lanczos on :math:`(R^\dagger R + \mu I)^{-1}`, over a
  window of :data:`WINDOW` times the values kept, applied
  through one sparse LU of the augmented system

  .. math::

     \begin{pmatrix} I & R \\ R^\dagger & -\mu I \end{pmatrix}
     \begin{pmatrix} r \\ x \end{pmatrix} = \begin{pmatrix} 0 \\ c \end{pmatrix},
     \qquad x = -(R^\dagger R + \mu I)^{-1}c,

  so that :math:`R` is never squared: the normal matrix squares the condition number, and its
  rank-one constraint rows fill a dense block. A Rayleigh–Ritz step on :math:`R` itself, over the
  Lanczos basis, then gives the singular values to about :math:`\varepsilon\sigma_{\max}`. The
  shift :math:`\mu = (10^{-9}\sigma_{\max})^2` keeps an exact kernel's system non-singular.
* **The largest.** Lanczos on :math:`R^\dagger R` itself, where squaring loses nothing at the
  top of the spectrum.
* **The ground state's bound,** from :math:`R`: :math:`\sigma_0 \le \|Rv_0\|`, and the ground
  state is within :math:`1 - |\langle v_0, v_\star\rangle|^2 \le \|Rv_0\|^2/(\sigma_1^2 -
  \|Rv_0\|^2)` of the exact one (:attr:`~pihm.classical.reference.ReferenceState.fidelity_bound`).
  The sharper lemma :math:`1-|\langle v_0,v_\star\rangle|^2 \le (r^2-\sigma_0^2)/
  (\sigma_1^2-\sigma_0^2)` needs
  :math:`\sigma_0`, which is not known here, so this looser form is the one reported. It takes
  :math:`\sigma_1` as Lanczos found it, which is exact to rounding once converged and provided no
  singular value below it was missed; it is an a posteriori estimate, not a guarantee.
  At the exact ground state it is :math:`\sigma_0^2/(\sigma_1^2-\sigma_0^2)`, so it is sharp where
  :math:`R` has a kernel and floors at about :math:`(\sigma_0/\sigma_1)^2` where it has none (a
  Carleman lift, whose truncation leaves :math:`\sigma_0>0`).

Both Lanczos runs start from a Gaussian vector of a fixed seed (:func:`start_vector`),
answer-independent and deterministic. The smallest :math:`k + 1` singular values are found,
:math:`k` the largest kernel the detector considers
(:func:`~pihm.classical.reference.locate_kernel`), so a trusted state means the same here as from
the dense SVD.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh, splu

from pihm.classical.reference import (
    DEFAULT_MIN_GAP_RATIO,
    ReferenceState,
    fix_phase,
    locate_kernel,
)
from pihm.operators.tensor import SparseArray

__all__ = [
    "DENSE_COLUMNS",
    "START_SEED",
    "WINDOW",
    "iterative_reference_state",
    "largest_singular_value",
    "start_vector",
]

#: The most physical columns the dense SVD takes, a state of 4 x 1024 on one axis and 8 x 1024 on
#: two: at most a few gigabytes, and about 5 minutes at 7,300 columns. Past it the
#: reference is iterative.
DENSE_COLUMNS: Final[int] = 8192

#: The seed of the Lanczos runs' start vector (:func:`start_vector`).
START_SEED: Final[int] = 0

#: How many times the singular values it keeps the shift-invert Lanczos seeks. Lanczos from one
#: vector meets the further copies of a repeated singular value through rounding alone, so a
#: degenerate cluster cut by the window's edge is found incompletely (R10's spectra repeat a value
#: ten times); twice the window puts every cluster of up to the kept count inside it.
WINDOW: Final[int] = 2

#: The shift :math:`\mu` relative to :math:`\sigma_{\max}^2`.
_SHIFT: Final[float] = 1.0e-18

#: Lanczos's tolerance on :math:`\sigma_{\max}^2`, relative.
_TOLERANCE: Final[float] = 1.0e-12


def _sparse(R: NDArray[Any] | SparseArray) -> sp.csc_array:
    """Return ``R`` as a sparse float64 or complex128 array, refusing lower precision."""
    matrix = sp.csc_array(R)
    if matrix.dtype not in (np.float64, np.complex128):
        raise TypeError(f"the residual must be float64 or complex128, got {matrix.dtype}")
    return matrix


def start_vector(size: int, dtype: np.dtype[Any] | type) -> NDArray[Any]:
    """Return the Lanczos runs' start: a Gaussian vector of :data:`START_SEED`.

    It has a component on every singular vector, as a start must for Lanczos to meet every
    singular value. A structured start need not: a residual that commutes with a permutation of
    its columns keeps the uniform vector in one symmetry sector, and Lanczos from it never meets
    the others' singular values (on R10's Taylor–Green vortex a uniform start missed 9 of the
    lowest 17 in our reference runs; no test asserts this).
    """
    vector: NDArray[Any] = np.random.default_rng(START_SEED).standard_normal(size).astype(dtype)
    return vector


def largest_singular_value(R: NDArray[Any] | SparseArray) -> float:
    r"""Return :math:`\sigma_{\max}(R)`, by Lanczos on :math:`R^\dagger R` (:func:`start_vector`).

    Squaring loses nothing at the top of the spectrum.
    """
    matrix = _sparse(R)
    columns = matrix.shape[1]
    adjoint = matrix.conj().T

    def normal(v: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(adjoint @ (matrix @ v))

    operator = LinearOperator((columns, columns), matvec=normal, dtype=matrix.dtype)
    try:
        (value,) = eigsh(
            operator,
            k=1,
            which="LA",
            ncv=min(columns - 1, 24),
            tol=_TOLERANCE,
            v0=start_vector(columns, matrix.dtype),
            return_eigenvectors=False,
        )
    except ArpackNoConvergence as failure:
        if not len(failure.eigenvalues):
            raise
        (value,) = failure.eigenvalues[:1]
    return float(np.sqrt(max(float(np.real(value)), 0.0)))


def iterative_reference_state(
    R: NDArray[Any] | SparseArray,
    *,
    min_gap_ratio: float = DEFAULT_MIN_GAP_RATIO,
    max_kernel: int = 16,
) -> ReferenceState:
    r"""Return the ground state of ``R`` and its smallest spectrum, iteratively (module docs).

    Parameters
    ----------
    R
        The stacked residual, sparse or dense, float64 or complex128, with more columns than
        ``max_kernel + 1``.
    min_gap_ratio, max_kernel
        As :func:`~pihm.classical.reference.reference_state` takes them.

    Returns
    -------
    ReferenceState
        Its ``singular_values`` the smallest ``max_kernel + 1`` and then :math:`\sigma_{\max}`,
        ``complete`` false, ``driver`` ``"iterative"`` and the ground state's bound.

    Raises
    ------
    ValueError
        For a residual with too few columns for the kernel search.
    """
    matrix = _sparse(R)
    rows, columns = matrix.shape
    count = max_kernel + 1
    if columns <= count:
        raise ValueError(f"an iterative reference needs more than {count} columns")
    sigma_max = largest_singular_value(matrix)
    adjoint = matrix.conj().T
    shift = _SHIFT * sigma_max**2
    augmented = sp.block_array(
        [[sp.eye_array(rows), matrix], [adjoint, -shift * sp.eye_array(columns)]], format="csc"
    )
    factors = splu(augmented, permc_spec="COLAMD")
    zeros = np.zeros(rows, dtype=matrix.dtype)

    def inverse(c: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(-factors.solve(np.concatenate([zeros, np.ravel(c)]))[rows:])

    operator = LinearOperator((columns, columns), matvec=inverse, dtype=matrix.dtype)
    window = min(WINDOW * count, columns - 1)
    _values, basis = eigsh(operator, k=window, which="LA", v0=start_vector(columns, matrix.dtype))
    left, singular, right = np.linalg.svd(matrix @ basis, full_matrices=False)
    order = np.argsort(singular)[:count]
    singular = singular[order]
    vectors = basis @ right.conj().T[:, order]
    v, _u = fix_phase(vectors[:, 0], left[:, order[0]])
    sigma = float(singular[0])
    image = matrix @ v
    # At a kernel the left vector is undetermined, so the backward error is the pair (sigma^2, v)'s
    # for R^dagger R, relative to its norm: a few eps for an accurate state, like the dense one.
    backward = float(np.linalg.norm(adjoint @ image - sigma**2 * v))
    kernel_dimension, gap_ratio = locate_kernel(singular, max_kernel)
    residual = float(np.linalg.norm(image))
    excited = float(singular[1])
    return ReferenceState(
        vector=v,
        singular_values=np.append(singular, sigma_max),
        kernel_dimension=kernel_dimension,
        gap_ratio=gap_ratio,
        backward_error=backward / sigma_max**2 if sigma_max > 0 else 0.0,
        driver="iterative",
        trusted=kernel_dimension == 1 and gap_ratio >= min_gap_ratio,
        complete=False,
        fidelity_bound=residual**2 / (excited**2 - residual**2) if excited > residual else np.inf,
    )
