r"""The classical reference: the SVD ground state of a stacked residual, and whether to trust it.

The ground state of :math:`H = R^\dagger R` is the smallest right singular vector of the stacked
residual :math:`R`. It is computed from :math:`R` directly: never from
``eigh(R^T R)``, which squares the condition number and fails for the standard form beyond
:math:`n \approx 7` (:math:`1 - |\cos| = 0.73` at :math:`n = 8`, against
:math:`10^{-16}` for the SVD). Spectral quantities follow from the singular values,
:math:`\lambda_i = \sigma_i^2`.

Three safeguards are built in:

* **Fallback.** LAPACK's divide-and-conquer driver ``gesdd`` can fail to converge on stacked
  operators; the QR-based ``gesvd`` is then used.
* **Precision.** Single-precision input is refused: float32 contamination anywhere caps the
  accuracy of everything computed from it.
* **Trust.** A one-dimensional kernel found "by luck" in an under-resolved system is not
  trusted. The kernel is located at the largest ratio between consecutive small singular
  values, and is trusted only when that ratio is large.

:func:`reference_state` is the dense SVD, the reference wherever it fits;
:mod:`pihm.classical.iterative` computes the same quantities past that ceiling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from pihm.operators.tensor import SparseArray

__all__ = [
    "DEFAULT_MIN_GAP_RATIO",
    "ReferenceState",
    "SvdDriver",
    "fix_phase",
    "gesdd_fits",
    "locate_kernel",
    "reference_state",
]

logger = logging.getLogger(__name__)

#: The smallest ratio :math:`\sigma_k/\sigma_{k-1}` at the kernel boundary for a result to be
#: trusted: the largest spacing the singular spectra of the ladder show without a kernel. The
#: detector takes the largest of the 16 ratios formed from the lowest 17 singular values, so a
#: spectrum with no kernel reports its largest ordinary spacing. Over the build records of R1-R8
#: that is at most 19.7 (the stiff R2c's structural :math:`\sigma_2/\sigma_1`), its median 2.6 and
#: its 95th percentile 11.2; the only larger second ratios are three faithful R5 panels' second
#: near-kernel, not a spacing.
#: A ratio above it cannot come from spacing alone. It measures how sharply the residual fixes
#: the ground state, whose field sits about :math:`\sigma_0/\sigma_1` from the resolved one; the
#: field's own accuracy is reported beside it
#: (:attr:`~pihm.classical.ClassicalSolution.field_error_l2`).
DEFAULT_MIN_GAP_RATIO: Final[float] = 20.0

#: How a reference was computed. The LAPACK SVD drivers, in the order they are tried, or the
#: iterative route past the dense ceiling (:mod:`pihm.classical.iterative`).
SvdDriver: TypeAlias = Literal["gesdd", "gesvd", "iterative"]

#: Singular values are floored here before ratios are taken, so an exact zero gives a finite,
#: huge ratio rather than a division by zero.
_FLOOR: Final[float] = 1.0e-300

#: The largest integer a 32-bit LAPACK holds, its workspace counts included.
_LAPACK32_MAX: Final[int] = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class ReferenceState:
    r"""The SVD ground state of a stacked residual, with its spectrum and diagnostics.

    Attributes
    ----------
    vector
        The unit smallest right singular vector. The phase is fixed so that its largest
        component is real and positive, which makes results deterministic.
    singular_values
        All singular values in ascending order, padded with zeros for a wide residual; an
        iterative reference's (:attr:`complete` false) are its smallest few and then
        :math:`\sigma_{\max}`, which is what every quantity below reads.
    kernel_dimension
        The number of singular values below the largest ratio gap among the smallest
        ``max_kernel`` of them.
    gap_ratio
        That largest ratio, :math:`\sigma_k/\sigma_{k-1}` at the kernel boundary.
    backward_error
        :math:`\max(\|Rv - \sigma u\|, \|R^\dagger u - \sigma v\|)/\|R\|_2` for the returned
        triplet, which should be a small multiple of machine precision. An iterative reference
        gives the pair's for :math:`H`, :math:`\|R^\dagger Rv - \sigma^2 v\|/\|R\|_2^2`, since at a
        kernel its left vector is undetermined.
    driver
        How the reference was computed: ``"gesdd"`` or ``"gesvd"`` (LAPACK's SVD), or
        ``"iterative"``.
    trusted
        Whether the kernel is one-dimensional and cleanly separated
        (``gap_ratio >= min_gap_ratio``).
    complete
        Whether :attr:`singular_values` holds the whole spectrum.
    fidelity_bound
        An iterative reference's a posteriori bound on its ground state, from :math:`R`:
        :math:`1 - |\langle v, v_\star\rangle|^2 \le \|Rv\|^2/(\sigma_1^2 - \|Rv\|^2)`, at least
        :math:`\sigma_0^2/(\sigma_1^2-\sigma_0^2)` however accurate :math:`v`.
        ``None`` for a dense one, whose backward stability :attr:`vector_error_bound` bounds.
    right_vectors
        A dense reference's right singular vectors, one per row, in descending order of
        :attr:`singular_values` (a wide residual's null space last): the whole spectral
        decomposition, which the error budget reads (:func:`pihm.qsp.budget.state_budget`).
        ``None`` for an iterative one.
    """

    vector: NDArray[Any]
    singular_values: NDArray[np.float64]
    kernel_dimension: int
    gap_ratio: float
    backward_error: float
    driver: SvdDriver
    trusted: bool
    complete: bool = True
    fidelity_bound: float | None = None
    right_vectors: NDArray[Any] | None = field(default=None, repr=False, compare=False)

    @property
    def lambda0(self) -> float:
        r""":math:`\lambda_0 = \sigma_0^2`, the ground energy of :math:`H = R^\dagger R`."""
        return float(self.singular_values[0] ** 2)

    @property
    def lambda1(self) -> float:
        r""":math:`\lambda_1 = \sigma_1^2`, the first excited energy."""
        return float(self.singular_values[1] ** 2)

    @property
    def gap(self) -> float:
        r"""The absolute gap :math:`\Delta = \lambda_1 - \lambda_0`."""
        return self.lambda1 - self.lambda0

    @property
    def hamiltonian_norm(self) -> float:
        r""":math:`\|H\|_2 = \sigma_{\max}^2`."""
        return float(self.singular_values[-1] ** 2)

    @property
    def relative_gap(self) -> float:
        r"""The relative gap :math:`\Delta / \|H\|_2`."""
        return self.gap / self.hamiltonian_norm

    @property
    def vector_error_bound(self) -> float:
        r"""Wedin's bound on the angle between the computed and the exact ground state.

        A backward-stable SVD perturbs :math:`R` by about :math:`\varepsilon\|R\|_2`, which
        tilts the smallest right singular vector by at most
        :math:`\varepsilon\sigma_{\max}/(\sigma_1 - \sigma_0)`. In the standard form
        :math:`\sigma_{\max}` grows like :math:`N^4`, so this is the floor of the reference
        itself at large :math:`n` (infinite for a degenerate kernel).
        """
        separation = float(self.singular_values[1] - self.singular_values[0])
        noise = float(np.finfo(np.float64).eps) * float(self.singular_values[-1])
        return noise / separation if separation > 0 else float("inf")

    def kernel_count(self, tolerance: float = 1.0e-10) -> int:
        r"""Count the singular values at most ``tolerance`` times :math:`\sigma_{\max}`.

        Unlike :attr:`kernel_dimension`, which looks for a gap among the smallest few, this
        counts a large degenerate kernel in full, where the
        spectrum is complete; an iterative reference counts among its few.
        """
        return int(np.count_nonzero(self.singular_values <= tolerance * self.singular_values[-1]))


def _dense(R: NDArray[Any] | SparseArray) -> NDArray[Any]:
    """Return ``R`` as a dense float64 or complex128 array, refusing lower precision."""
    matrix = R if isinstance(R, np.ndarray) else R.toarray()
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("the residual must be a non-empty matrix")
    if matrix.dtype not in (np.float64, np.complex128):
        raise TypeError(
            f"the residual must be float64 or complex128, got {matrix.dtype}; single precision "
            "caps the achievable accuracy"
        )
    return matrix


def gesdd_fits(shape: tuple[int, int]) -> bool:
    """Return whether gesdd's workspace for a matrix of ``shape`` fits a 32-bit LAPACK.

    The bound is the larger of gesdd's two largest workspace counts, real or complex,
    5n² + 7n and 2mn + 2n² + n, for the dimensions n ≤ m.

    Examples
    --------
    >>> gesdd_fits((34815, 19584)), gesdd_fits((34815, 29104))
    (True, False)
    """
    n, m = sorted(shape)
    return max(5 * n * n + 7 * n, 2 * m * n + 2 * n * n + n) <= _LAPACK32_MAX


def _svd(
    matrix: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[np.float64], NDArray[Any], SvdDriver]:
    """Return ``(U, s, Vh, driver)``, falling back from gesdd to gesvd.

    SciPy's LAPACK has 32-bit integers on Linux, NumPy's 64-bit ones; a matrix whose workspace a
    32-bit LAPACK cannot count (:func:`gesdd_fits`) goes to NumPy's gesdd.
    """
    full = matrix.shape[0] < matrix.shape[1]
    try:
        if not gesdd_fits(matrix.shape):
            U, s, Vh = np.linalg.svd(matrix, full_matrices=full)
            return U, s, Vh, "gesdd"
        U, s, Vh = scipy.linalg.svd(matrix, full_matrices=full, lapack_driver="gesdd")
        return U, s, Vh, "gesdd"
    except np.linalg.LinAlgError:
        logger.warning("gesdd did not converge on a %s residual; retrying with gesvd", matrix.shape)
        U, s, Vh = scipy.linalg.svd(matrix, full_matrices=full, lapack_driver="gesvd")
        return U, s, Vh, "gesvd"


def locate_kernel(ascending: NDArray[np.float64], max_kernel: int) -> tuple[int, float]:
    """Return the kernel's dimension and gap ratio: the largest ratio among the smallest values.

    ``ascending`` holds at least the smallest ``max_kernel + 1`` singular values in order.
    """
    floored = np.maximum(ascending, _FLOOR)
    limit = min(max_kernel, ascending.size - 1)
    if limit <= 0:
        return 0, float("inf")
    ratios = floored[1 : limit + 1] / floored[:limit]
    dimension = int(np.argmax(ratios)) + 1
    return dimension, float(ratios[dimension - 1])


def fix_phase(v: NDArray[Any], u: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return a singular pair with ``v``'s largest component real and positive."""
    peak = v[np.argmax(np.abs(v))]
    phase = peak / abs(peak)
    return v / phase, u / phase


def reference_state(
    R: NDArray[Any] | SparseArray,
    *,
    min_gap_ratio: float = DEFAULT_MIN_GAP_RATIO,
    max_kernel: int = 16,
) -> ReferenceState:
    r"""Return the SVD ground state of the stacked residual ``R``.

    Parameters
    ----------
    R
        The stacked residual :math:`[R_1; R_2; \dots]`, dense or sparse, float64 or complex128.
        It is densified; this is the reference path, for sizes a dense SVD can handle.
    min_gap_ratio
        The ratio a one-dimensional kernel must clear to be trusted.
    max_kernel
        The largest kernel dimension considered when locating the gap.

    Returns
    -------
    ReferenceState
        The ground state and its diagnostics.

    Raises
    ------
    TypeError
        If ``R`` is not double precision.

    Examples
    --------
    >>> R = np.diag([3.0, 2.0, 1e-9])
    >>> state = reference_state(R)
    >>> state.vector
    array([0., 0., 1.])
    >>> state.kernel_dimension, state.trusted
    (1, True)
    """
    matrix = _dense(R)
    U, s, Vh, driver = _svd(matrix)
    columns = matrix.shape[1]
    ascending = np.concatenate([np.zeros(columns - s.size), s[::-1]])
    kernel_dimension, gap_ratio = locate_kernel(ascending, max_kernel)
    sigma = float(ascending[0])
    u = U[:, s.size - 1] if columns == s.size else np.zeros(matrix.shape[0], dtype=matrix.dtype)
    v, u = fix_phase(Vh[-1].conj(), u)
    norm = float(s[0]) if s.size else 0.0
    residual = max(
        float(np.linalg.norm(matrix @ v - sigma * u)),
        float(np.linalg.norm(matrix.conj().T @ u - sigma * v)),
    )
    return ReferenceState(
        vector=v,
        singular_values=ascending,
        kernel_dimension=kernel_dimension,
        gap_ratio=gap_ratio,
        backward_error=residual / norm if norm > 0 else 0.0,
        driver=driver,
        trusted=kernel_dimension == 1 and gap_ratio >= min_gap_ratio,
        right_vectors=Vh,
    )
