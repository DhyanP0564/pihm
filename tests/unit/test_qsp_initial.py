"""The initial states a preparation tries."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.assemble import assemble
from pihm.problems.ladder import problem
from pihm.qsp.initial import (
    field_weights,
    geometric_state,
    initial_states,
    random_state,
    uniform_state,
)


def test_geometric_state_is_normalised_and_matches_r_powers() -> None:
    state = geometric_state(5, 0.5)
    assert np.linalg.norm(state) == pytest.approx(1.0)
    ratios = state[1:] / state[:-1]
    assert ratios == pytest.approx(0.5, abs=1e-12)


def test_geometric_state_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="dimension"):
        geometric_state(0, 0.5)
    with pytest.raises(ValueError, match="nonzero"):
        geometric_state(3, 0.0)


def test_uniform_state_is_normalised_and_flat() -> None:
    state = uniform_state(8)
    assert np.linalg.norm(state) == pytest.approx(1.0)
    assert np.allclose(np.abs(state), 1.0 / np.sqrt(8))


def test_random_state_is_normalised_and_reproducible_from_a_seed() -> None:
    a = random_state(6, np.random.default_rng(0))
    b = random_state(6, np.random.default_rng(0))
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    ("name", "order", "weights"),
    [
        ("R7", None, {0: 0.0, 1: 1.0}),
        ("R8", 3, {0: 1.0}),
        ("R10a-M12", 2, {4: 0.5**0.5, 5: 0.0, 6: -(0.5**0.5)}),
    ],
)
def test_a_systems_initial_state_lies_on_its_own_fields_weighted_by_the_data(
    name: str, order: int | None, weights: dict[int, float]
) -> None:
    """Each field's block carries the state times w0/|w0|; the lift's levels, nothing.

    R7 starts at (u, v) = (0, pi), so only v's block; Taylor-Green's c_1_m1 and c_1_1 (the fifth
    and seventh fields) carry +1 and -1, over sqrt 2.
    """
    residual = assemble(problem(name), 3, regime="descriptor", carleman=order)
    (state,) = initial_states("geometric", residual, (0.5,))
    size = 8
    fields = residual.field_blocks
    data = field_weights(residual)
    for block in range(len(residual.derivatives)):
        part = state[block * size : (block + 1) * size]
        weight = data[fields.index(block)] if block in fields else 0.0
        np.testing.assert_allclose(part, weight * geometric_state(size, 0.5), atol=1e-15)
    for field, weight in weights.items():
        assert data[field] == pytest.approx(weight, abs=1e-15)
    assert np.linalg.norm(state) == pytest.approx(1.0)
