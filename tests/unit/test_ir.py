"""The descriptor's term IR."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pihm.assemble import Rescale, assemble
from pihm.assemble.systems import coupling
from pihm.ir import Band, band_terms, matching_terms, system_ir, system_matrix, system_rows
from pihm.operators import banded_factors, multiplication
from pihm.problems import ladder

#: Every linear panel of the ladder, each variant, at small resolutions.
LINEAR = [
    (name, variant, n)
    for name in ("R1", "R2a", "R2b", "R2c", "R3a", "R3b", "R3c", "R3d", "R4a", "R4b", "R4c", "R4d")
    for variant in (("corrected",) if name == "R1" else ("faithful", "corrected"))
    for n in (2, 3, 5)
] + [
    (name, variant, n)
    for name in ("R5a", "R5b", "R5c")
    for variant in ("faithful", "corrected")
    for n in ((2, 2), (3, 2))
]


def _sum_of(terms: list[tuple[float, Band]], n_in: int, n_out: int) -> np.ndarray:
    total = np.zeros((1 << n_out, 1 << n_in))
    for coefficient, band in terms:
        total += coefficient * band.matrix(n_in, n_out).toarray()
    return total


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_band_terms_reconstruct_the_recurrences_factors(n: int) -> None:
    """B~ (constant bands) and D~ (a band affine in the index) as shift x diagonal terms."""
    B, D = banded_factors(n)
    for matrix in (B, D, -D):
        np.testing.assert_allclose(
            _sum_of(band_terms(matrix, n, n), n, n), matrix.toarray(), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("coefficients", [(0.0, 1.0), (1.0, 0.0, -1.0), (1.0, 0.0, -2.0, 0.0, 1.0)])
def test_band_terms_reconstruct_multiplications_square_and_lifted(
    coefficients: tuple[float, ...],
) -> None:
    for n in (3, 5):
        square = multiplication(coefficients, n, mode="square")
        np.testing.assert_allclose(
            _sum_of(band_terms(square, n, n), n, n), square.toarray(), rtol=0, atol=1e-12
        )
        lifted = multiplication(coefficients, n, mode="lift")
        np.testing.assert_allclose(
            _sum_of(band_terms(lifted, n, n + 1), n, n + 1), lifted.toarray(), rtol=0, atol=1e-12
        )


def test_an_unstructured_band_is_refused() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="off its affine fit"):
        band_terms(sp.diags_array(rng.normal(size=64)), 6, 6)


@pytest.mark.parametrize("rescale", ["block", "none"])
@pytest.mark.parametrize(("name", "variant", "n"), LINEAR)
def test_the_ir_sums_to_the_assembled_system_rows(
    name: str, variant: str, n: int | tuple[int, ...], rescale: Rescale
) -> None:
    """IR mode: the terms a circuit implements equal the assembled rows."""
    residual = assemble(ladder.problem(name, variant), n, regime="descriptor", rescale=rescale)
    A = system_matrix(residual)
    B = system_ir(residual).to_sparse()
    scale = float(np.max(np.abs(A.data)))
    assert A.shape == B.shape
    np.testing.assert_allclose(B.toarray(), A.toarray(), rtol=0, atol=1e-12 * scale)


def test_each_system_row_block_sits_at_a_field_index_of_its_own() -> None:
    """R2a: the recurrences at their lower blocks, the equation at the top, the padding at 3."""
    residual = assemble(ladder.problem("R2a"), 3, regime="descriptor")
    assert sorted(system_rows(residual)) == [
        (0, "recurrence f^(0) -> f^(1) along x"),
        (1, "recurrence f^(1) -> f^(2) along x"),
        (2, "equation"),
        (3, "padding block 3"),
    ]
    laplace = assemble(ladder.problem("R5a"), 2, regime="descriptor")
    indices = [at for at, _ in system_rows(laplace)]
    assert sorted(indices) == list(range(len(laplace.derivatives)))


def test_the_recurrences_share_their_derivative_term() -> None:
    """D~, the largest term, is paid for once across the recurrences: alpha -> ||A_sys||."""
    residual = assemble(ladder.problem("R2a"), 8, regime="descriptor", rescale="none")
    A = system_matrix(residual)
    norm = spla.svds(A, k=1, return_singular_vectors=False, v0=np.ones(A.shape[1]))[0]
    assert system_ir(residual).alpha <= 1.05 * norm


def test_the_ir_takes_linear_descriptor_residuals_only() -> None:
    with pytest.raises(ValueError, match="descriptor residuals"):
        system_ir(assemble(ladder.problem("R2a"), 3))


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_matchings_sum_to_the_matrix_and_their_permutations_reach_their_rows(width: int) -> None:
    rng = np.random.default_rng(width)
    F = rng.normal(size=(1 << width, 1 << width)) * (rng.random((1 << width,) * 2) < 0.4)
    terms = matching_terms(F)
    total = sum((t.coefficient * t.field_matrix(width).toarray() for t in terms), np.zeros_like(F))
    np.testing.assert_allclose(total, F, atol=1e-12)
    for term in terms:
        assert term.matching is not None
        assert max(abs(v) for v in term.matching.values) == pytest.approx(1.0)
        images = term.matching.permutation(width)
        assert sorted(images) == list(range(1 << width))
        pairs = zip(term.matching.columns, term.matching.rows, strict=True)
        assert all(images[c] == r for c, r in pairs)


def test_a_carleman_lifts_matchings_are_as_many_as_its_largest_degree() -> None:
    """R9c-M4's lift at K = 3, seven matchings against 1061 Pauli words."""
    A = coupling(ladder.problem("R9c-M4"), 3)
    degree = max(int(np.max(np.diff(A.tocsr().indptr))), int(np.max(np.diff(A.tocsc().indptr))))
    terms = matching_terms(A.toarray())
    assert len(terms) == degree == 7
    assert sum(abs(t.coefficient) for t in terms) < 1.3 * np.linalg.norm(A.toarray(), 2)


def test_a_matching_of_one_signed_value_carries_its_sign_in_the_coefficient() -> None:
    (term,) = matching_terms(np.array([[-2.0]]))
    assert term.coefficient == -2.0
    assert term.matching is not None
    assert term.matching.is_permutation(0)
