"""Checks of the interferometric readout."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.basis import tau
from pihm.circuits.assemble import encode_standard_linear
from pihm.problems.ladder import problem
from pihm.qsp.feasibility import Budget, InfeasibleError
from pihm.qsp.prepare import t2
from pihm.qsp.readout import combined_statevectors, feature_amplitudes, readout


@pytest.mark.parametrize("x", [-0.9, -0.2, 0.5, 1.0])
def test_feature_amplitudes_are_tau_over_sqrt2(x: float) -> None:
    np.testing.assert_allclose(feature_amplitudes(x, 3), tau(x, 3) / np.sqrt(2.0), atol=1e-12)


def test_the_combiner_holds_the_prepared_state_and_the_combined_state() -> None:
    # G's all-ancilla-zero branch is sqrt(p_G) psi (T2's state, rescaled by the phases' own
    # rescale); C's is (e_0 + g)/2.
    p = problem("R1")
    stateprep = t2(p, 2)
    stacked, _reflection = encode_standard_linear(p, 2)
    g_state, c_state = combined_statevectors(stateprep, stacked, 2)
    g = g_state[:4]
    direction = g / np.linalg.norm(g)
    np.testing.assert_allclose(direction, stateprep.state, atol=1e-12)
    e0 = np.eye(4)[0]
    np.testing.assert_allclose(c_state[:4], (e0 + g) / 2.0, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_t1_reads_out_what_t2_does(n: int) -> None:
    # The combination is an identity, not an approximation: T1 with exact probabilities equals
    # the exact overlap to rounding, field and eta alike, although p_G < 1.
    p = problem("R1")
    exact = readout(p, n, tier="T2")
    circuit = readout(p, n, tier="T1")
    assert circuit.prepared_probability < 0.95
    scale = np.max(np.abs(exact.field))
    assert np.max(np.abs(circuit.field - exact.field)) / scale < 1e-10
    assert abs(circuit.eta - exact.eta) / exact.eta < 1e-10


def test_t2_recovers_the_field_and_its_own_regular_datum() -> None:
    result = readout(problem("R1"), 3, tier="T2")
    assert result.points[-1] == problem("R1").regular.at[0]
    assert result.field[-1] == pytest.approx(result.reference[-1])
    assert np.max(np.abs(result.field - result.reference)) < 1e-4


def test_sampled_probabilities_scatter_about_the_exact_field() -> None:
    p = problem("R1")
    result = readout(p, 2, tier="T1", shots=200_000, rng=np.random.default_rng(1))
    departure = np.max(np.abs(result.sampled_field - result.field)) / np.max(np.abs(result.field))
    assert 0.0 < departure < 0.1
    assert np.isnan(readout(p, 2, tier="T1").sampled_eta)


def test_the_readout_refuses_a_run_over_its_budget() -> None:
    with pytest.raises(InfeasibleError, match="memory"):
        readout(problem("R1"), 2, tier="T1", budget=Budget(1e6, 3600.0, "override"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tier": "T1", "shots": -1}, "non-negative"),
        ({"tier": "T2", "shots": 10}, "tier T1"),
        ({"tier": "T1", "shots": 10}, "random generator"),
    ],
)
def test_the_readout_rejects_inconsistent_shots(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        readout(problem("R1"), 2, **kwargs)  # type: ignore[arg-type]


def test_t2_reads_a_field_on_two_axes_out_on_their_grid() -> None:
    """Two axes: the probe grid is READOUT_POINTS squared, the regular point last."""
    p = problem("R5a", "corrected")
    result = readout(p, (2, 2), tier="T2")
    assert result.points.shape == (50, 2)
    np.testing.assert_allclose(result.points[-1], p.regular.at)
    features = np.array([np.kron(tau(x, 2), tau(y, 2)) for x, y in result.points])
    overlaps = features @ result.stateprep.state
    expected = result.reference[-1] * np.real(overlaps / overlaps[-1])
    np.testing.assert_allclose(result.field, expected, atol=1e-12)


@pytest.mark.slow
def test_t1_reads_out_what_t2_does_on_two_axes() -> None:
    """The combiner on a two-axis field (R5b, n = (2, 2)): the probabilities carry 2/2^d."""
    p = problem("R5b", "corrected")
    stateprep = t2(p, (2, 2))
    stacked, _ = encode_standard_linear(p, (2, 2))
    g_state, _c_state = combined_statevectors(stateprep, stacked, 4)
    g = g_state[:16]
    expected = stateprep.state * np.sqrt(np.vdot(g, g).real)
    phase = np.vdot(expected, g) / abs(np.vdot(expected, g))
    np.testing.assert_allclose(g, phase * expected, atol=1e-10)
