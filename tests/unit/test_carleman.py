"""The Carleman lift, checked against its definition."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.carleman import (
    carleman_dimension,
    carleman_matrix,
    moments,
    monomials,
    nonlinearity_ratio,
    symmetric_embedding,
    tensor_matrix,
    tensor_moments,
    tensor_terms,
)


def _field_of_two() -> list[sp.csr_array]:
    """Return a two-component quadratic field with every kind of entry, F_2 asymmetric."""
    linear = sp.csr_array([[-1.0, 0.5], [0.25, -2.0]])
    quadratic = np.zeros((2, 4))
    quadratic[0, 1] = 0.7  # u_0 u_1, written as (0, 1)
    quadratic[0, 2] = -0.3  # u_1 u_0, the same monomial written as (1, 0)
    quadratic[1, 0] = 1.1  # u_0^2
    quadratic[1, 3] = -0.4  # u_1^2
    return [linear, sp.csr_array(quadratic)]


@pytest.mark.parametrize(("fields", "order"), [(1, 1), (1, 5), (2, 3), (3, 4)])
def test_the_monomials_run_level_by_level_with_the_fields_first(fields: int, order: int) -> None:
    keys = monomials(fields, order)
    assert len(keys) == len(set(keys)) == carleman_dimension(fields, order)
    assert [sum(key) for key in keys] == sorted(sum(key) for key in keys)
    assert keys[:fields] == tuple(tuple(int(i == j) for i in range(fields)) for j in range(fields))
    assert all(1 <= sum(key) <= order and min(key) >= 0 for key in keys)


def test_a_scalar_quadratic_lifts_to_a_bidiagonal_matrix() -> None:
    """The field y' = a y + b y^2 gives (y^k)' = k a y^k + k b y^(k+1), cut at level K."""
    a, b, order = -1.5, 0.75, 5
    matrix = carleman_matrix([sp.csr_array([[a]]), sp.csr_array([[b]])], order).toarray()
    levels = np.arange(1, order + 1)
    expected = np.diag(levels * a) + np.diag(levels[:-1] * b, 1)
    np.testing.assert_array_equal(matrix, expected)


def test_the_lift_is_exact_below_the_truncation() -> None:
    """For |alpha| <= K - 1 no truncated monomial enters: dz_alpha/dt = (A_K z)_alpha exactly."""
    field = _field_of_two()
    order = 4
    keys = monomials(2, order)
    matrix = carleman_matrix(field, order)
    rng = np.random.default_rng(7)
    for _ in range(5):
        u = rng.normal(size=2)
        du = field[0] @ u + field[1] @ np.kron(u, u)
        z = moments(u, keys)
        # d/dt prod u_i^a_i = sum_i a_i u^(a - e_i) du_i
        dz = np.array(
            [
                sum(
                    key[i] * np.prod(u ** (np.asarray(key) - np.eye(2, dtype=int)[i])) * du[i]
                    for i in range(2)
                    if key[i]
                )
                for key in keys
            ]
        )
        exact = [index for index, key in enumerate(keys) if sum(key) <= order - 1]
        np.testing.assert_allclose((matrix @ z)[exact], dz[exact], rtol=1e-12, atol=1e-12)


def test_a_linear_field_is_its_own_lift() -> None:
    field = _field_of_two()
    np.testing.assert_array_equal(carleman_matrix(field[:1], 1).toarray(), field[0].toarray())
    np.testing.assert_array_equal(carleman_matrix(field, 1).toarray(), field[0].toarray())


def test_the_moments_are_the_monomials_of_the_data() -> None:
    keys = monomials(2, 2)
    np.testing.assert_allclose(moments([2.0, 3.0], keys), [2.0, 3.0, 4.0, 6.0, 9.0])


def test_the_nonlinearity_ratio() -> None:
    """R8's R = |y_0| |F_2| / |Re lambda_1(F_1)| = 1/2, independent of the time scale."""
    for scale in (1.0, 1.5):
        field = [sp.csr_array([[-scale]]), sp.csr_array([[scale]])]
        assert nonlinearity_ratio(field, [0.5]) == pytest.approx(0.5, rel=1e-15)
    assert nonlinearity_ratio(_field_of_two()[:1], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="damped"):
        nonlinearity_ratio([sp.csr_array([[1.0]]), sp.csr_array([[1.0]])], [0.5])


def test_malformed_fields_are_refused() -> None:
    with pytest.raises(ValueError, match="at least"):
        monomials(0, 2)
    with pytest.raises(ValueError, match="linear part"):
        carleman_matrix([], 2)
    with pytest.raises(ValueError, match="shape"):
        carleman_matrix([sp.csr_array(np.eye(2)), sp.csr_array(np.ones((2, 2)))], 2)


# ---------------------------------------------------------------------------- the tensor layout


@pytest.mark.parametrize("order", [1, 2, 3, 6])
def test_the_tensor_layout_has_one_term_per_degree_and_first_slot(order: int) -> None:
    """2K - 1 for a quadratic field, whatever the number of fields."""
    terms = tensor_terms(order)
    assert len(terms) == 2 * order - 1
    assert len(tensor_terms(order, (1,))) == order
    assert all(term.slot + term.degree <= order for term in terms)


def test_the_tensor_layout_is_the_kronecker_sum_level_by_level() -> None:
    """The level blocks of A_3: F1, F2; F1 x I + I x F1, F2 x I + I x F2; and so on."""
    F1, F2 = _field_of_two()
    matrix = tensor_matrix([F1, F2], 3).toarray()
    I2 = np.eye(2)
    F1_, F2_ = F1.toarray(), F2.toarray()
    np.testing.assert_array_equal(matrix[0:2, 0:2], F1_)
    np.testing.assert_array_equal(matrix[0:2, 2:6], F2_)
    np.testing.assert_allclose(matrix[2:6, 2:6], np.kron(F1_, I2) + np.kron(I2, F1_))
    np.testing.assert_allclose(matrix[2:6, 6:14], np.kron(F2_, I2) + np.kron(I2, F2_))
    np.testing.assert_allclose(
        matrix[6:14, 6:14],
        np.kron(np.kron(F1_, I2), I2)
        + np.kron(np.kron(I2, F1_), I2)
        + np.kron(I2, np.kron(I2, F1_)),
    )
    assert not matrix[2:, :2].any()
    assert not matrix[6:, :6].any()


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_the_tensor_layout_maps_the_symmetric_image_as_the_symmetric_layout(order: int) -> None:
    """A_tensor V = V A_K, on every vector, with F_2 written asymmetrically."""
    field = _field_of_two()
    V = symmetric_embedding(2, order)
    left = tensor_matrix(field, order) @ V
    right = V @ carleman_matrix(field, order)
    np.testing.assert_allclose(left.toarray(), right.toarray(), atol=1e-15)
    u = np.array([0.3, -0.7])
    np.testing.assert_allclose(V @ moments(u, monomials(2, order)), tensor_moments(u, order))
