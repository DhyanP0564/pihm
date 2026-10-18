"""Checks of the SU(2) nonlinear Fourier transform: forward/inverse round trip, both methods."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pihm.qsp.nlft import _LPoly, inlft_fast, inlft_layer_stripping, nlft


def _random_sequence(
    rng: np.random.Generator, degree: int, scale: float = 0.3
) -> NDArray[np.complex128]:
    F = rng.normal(size=degree + 1) + 1j * rng.normal(size=degree + 1)
    result: NDArray[np.complex128] = F * (scale / np.max(np.abs(F)))
    return result


@pytest.mark.parametrize("degree", [0, 1, 5, 30])
def test_inverse_nlft_recovers_the_forward_transform_both_methods(degree: int) -> None:
    rng = np.random.default_rng(degree)
    F = _random_sequence(rng, degree)
    a, b = nlft(F)
    assert np.max(np.abs(inlft_fast(a, b) - F)) < 1e-12
    assert np.max(np.abs(inlft_layer_stripping(a, b) - F)) < 1e-12


def test_fast_and_layer_stripping_agree_on_a_larger_random_sequence() -> None:
    rng = np.random.default_rng(1234)
    F = _random_sequence(rng, degree=200)
    a, b = nlft(F)
    assert np.max(np.abs(inlft_fast(a, b) - inlft_layer_stripping(a, b))) < 1e-10


def test_inlft_rejects_malformed_input() -> None:
    a = _LPoly([1.0], support_start=0)  # top exponent 0, but too short relative to a real b
    b = _LPoly([1.0, 2.0], support_start=0)
    with pytest.raises(ValueError, match="equally long"):
        inlft_fast(a, b)
    a_wrong_support = _LPoly([1.0, 2.0], support_start=-2)  # top exponent -1, not 0
    b2 = _LPoly([1.0, 2.0], support_start=0)
    with pytest.raises(ValueError, match="support"):
        inlft_fast(a_wrong_support, b2)
