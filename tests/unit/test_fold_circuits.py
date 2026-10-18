"""The product fold as a circuit, against pihm.operators.fold."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.circuits.folds import fold_encoding
from pihm.circuits.primitives import top_left_block
from pihm.operators import fold


@pytest.mark.parametrize("weight", ["none", "x"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_fold_encoding_is_the_fold_above_zeros(n: int, weight: str) -> None:
    """[N; 0] on the two copies' 2n qubits, for N_1 and N_x, at alpha = 2."""
    E = fold_encoding(n, weight)  # type: ignore[arg-type]
    target = fold(2, n, weight=weight).toarray()  # type: ignore[arg-type]
    padded = np.zeros((1 << 2 * n, 1 << 2 * n))
    padded[: target.shape[0]] = target
    assert E.alpha == 2.0
    np.testing.assert_allclose(top_left_block(E), padded, atol=1e-13)


def test_fold_encoding_ancilla() -> None:
    """Two padding qubits and n - 1 spares, and one more for the weight x."""
    assert fold_encoding(3).n_anc_qubits == 4
    assert fold_encoding(3, "x").n_anc_qubits == 5


def test_fold_encoding_refuses_a_bad_weight() -> None:
    """The weight is the plain product or the product times x."""
    with pytest.raises(ValueError, match="weight"):
        fold_encoding(2, "y")  # type: ignore[arg-type]
