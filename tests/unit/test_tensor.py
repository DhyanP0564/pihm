"""Multi-axis conventions: Kronecker order with axis 0 most significant."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.operators import axis_sizes, embed_axis, kron_all


@pytest.mark.gotcha
@pytest.mark.parametrize("ns", [(1, 2), (2, 1), (2, 3)])
def test_embed_axis_acts_on_the_named_axis(ns: tuple[int, int], rng: np.random.Generator) -> None:
    """An asymmetric, non-square operator on axis 0 or 1 of a separable field acts on that axis.

    earlier code once mixed the circuit convention (axis 0 low) with the Kronecker one
    (axis 0 high). With unequal axes and a lift that changes one axis's size, any mix-up fails.
    """
    b0, b1 = rng.normal(size=2 ** ns[0]), rng.normal(size=2 ** ns[1])
    A0 = sp.csr_array(rng.normal(size=(2 ** (ns[0] + 1), 2 ** ns[0])))
    A1 = sp.csr_array(rng.normal(size=(2 ** (ns[1] + 1), 2 ** ns[1])))
    field = np.kron(b0, b1)
    np.testing.assert_allclose(embed_axis(A0, ns, 0) @ field, np.kron(A0 @ b0, b1), atol=1e-12)
    np.testing.assert_allclose(embed_axis(A1, ns, 1) @ field, np.kron(b0, A1 @ b1), atol=1e-12)


def test_embed_axis_validates_its_input() -> None:
    with pytest.raises(ValueError, match="axis"):
        embed_axis(sp.eye_array(2), (1, 1), 2)
    with pytest.raises(ValueError, match="columns"):
        embed_axis(sp.eye_array(4), (1, 1), 0)


def test_kron_all_puts_the_first_factor_first() -> None:
    a = sp.csr_array(np.array([[1.0, 2.0]]))
    b = sp.csr_array(np.array([[1.0], [10.0]]))
    np.testing.assert_array_equal(kron_all([a, b]).toarray(), np.kron(a.toarray(), b.toarray()))
    with pytest.raises(ValueError, match="at least one"):
        kron_all([])


def test_axis_sizes() -> None:
    assert axis_sizes(3) == (8,)
    assert axis_sizes((1, 2)) == (2, 4)
    with pytest.raises(ValueError, match="non-negative"):
        axis_sizes((1, -1))
