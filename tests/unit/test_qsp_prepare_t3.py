"""Checks of tier T3 (resource estimation)."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.circuits.assemble import encode_standard_linear
from pihm.problems.ladder import problem
from pihm.qsp.prepare import t2, t3


def test_t3_predicted_probability_is_physically_valid() -> None:
    # The bug this guards against: a naive power-law extrapolation of gamma (a saturating,
    # bounded quantity) can predict gamma > 1 once evaluated past the sample range, giving a
    # predicted_p > 1 -- not a valid probability. Checked at an n well beyond the samples.
    result = t3(problem("R1"), 8, sample_n=(2, 3, 4))
    assert 0.0 <= result.predicted_p <= 1.0
    assert 0.0 <= result.gamma <= 1.0


def test_t3_gamma_is_taken_from_the_largest_sample_not_extrapolated() -> None:
    # gamma is taken from T2 at the LARGEST feasible n -- confirmed
    # directly that this means exactly t2's own measured value there, not a fitted/extrapolated
    # one (R1's own gamma saturates from n = 3 on, so a naive fit would overshoot).
    r_direct = t2(problem("R1"), 4)
    result = t3(problem("R1"), 8, sample_n=(2, 3, 4))
    assert result.gamma == pytest.approx(np.sqrt(r_direct.success_probability))


def test_t3_alpha_r_and_gap_extrapolate_close_to_a_direct_measurement() -> None:
    # Evaluating one step beyond the sample range should land close to what t2 itself measures
    # there (R1's own alpha_R grows smoothly with n; not an exact match, since the fit uses only
    # the samples, but within a few per cent).
    result = t3(problem("R1"), 5, sample_n=(2, 3, 4))
    direct = t2(problem("R1"), 5)
    assert result.alpha_r == pytest.approx(direct.alpha_r, rel=0.05)
    assert result.gap == pytest.approx(direct.reference.gap, rel=0.05)


def test_t3_amplification_rounds_and_gate_count_are_positive() -> None:
    result = t3(problem("R1"), 8, sample_n=(2, 3, 4))
    assert result.amplification_rounds >= 0.0
    assert result.preparations == 2.0 * result.amplification_rounds + 1.0
    assert result.gate_count > 0.0
    assert result.degree > 0
    assert result.n_qubits > 0


def test_t3_n_qubits_matches_the_actual_circuit_structure() -> None:
    from pihm.circuits.assemble import encode_standard_linear
    from pihm.circuits.compose import walk

    n = 3
    _stacked, reflection = encode_standard_linear(problem("R1"), n)
    expected = walk(reflection).n_qubits + 1
    result = t3(problem("R1"), n, sample_n=(2, 3, 4))
    assert result.n_qubits == expected


def test_t3_fits_report_one_entry_per_quantity() -> None:
    result = t3(problem("R1"), 8, sample_n=(2, 3, 4))
    assert set(result.fits) == {"alpha_r", "gap", "gamma", "gate_count"}
    for coefficient, _exponent in result.fits.values():
        assert coefficient > 0.0


def test_t3_rejects_too_few_sample_points() -> None:
    with pytest.raises(ValueError, match="sample_n"):
        t3(problem("R1"), 5, sample_n=(3,))


def test_direct_t3_measures_alpha_and_the_gap_at_n_itself() -> None:
    """Direct: alpha_R and the gap are t2's own at n; p is the tau -> infinity limit of t2's."""
    direct = t3(problem("R3c", "corrected"), 3)
    measured = t2(problem("R3c", "corrected"), 3)
    assert direct.method == "direct"
    assert direct.fits == {}
    assert direct.alpha_r == measured.alpha_r
    assert direct.gap == measured.gap
    assert direct.predicted_p == pytest.approx(measured.success_probability, rel=1e-5)


def test_direct_t3_reaches_a_printed_resolution_beyond_the_filter_ceiling() -> None:
    """R5c at its printed n = (5, 5): a degree past 2^20, estimated rather than refused."""
    result = t3(problem("R5c", "faithful"), (5, 5))
    assert result.degree > 1 << 20
    assert result.n == (5, 5)
    assert 0.0 < result.predicted_p <= 1.0


def test_direct_t3_counts_the_doubled_space_s_cited_transforms() -> None:
    """R6's walk holds the fold's dense transforms, counted apart from its cx."""
    result = t3(problem("R6a", "corrected"), 2)
    assert result.cited_count > 0.0
    assert t3(problem("R2a", "corrected"), 2).cited_count == 0.0


def test_t3_counts_amplitude_amplification_in_whole_preparations() -> None:
    """Amplification counts floor(pi / (4 arcsin gamma)) iterations, 2k + 1 preparations."""
    small = t3(problem("R3b", "corrected"), 3)  # p ~ 0.056
    large = t3(problem("R6a", "corrected"), 2)  # p ~ 0.86: one preparation suffices
    assert small.amplification_rounds == np.floor(np.pi / (4 * np.arcsin(small.gamma)))
    assert small.preparations == 2 * small.amplification_rounds + 1
    assert large.amplification_rounds == 0.0
    assert large.preparations == 1.0


def test_direct_t3_estimates_the_descriptor_form_at_n_itself() -> None:
    p = problem("R2a", "corrected")
    estimate = t3(p, 4, regime="descriptor")
    direct = t2(p, 4, regime="descriptor")
    assert estimate.alpha_r == pytest.approx(direct.alpha_r)
    assert estimate.gap == pytest.approx(direct.gap)
    assert estimate.gamma == pytest.approx(direct.gamma)


def test_direct_t3_gives_the_minimax_filter_s_exact_degree() -> None:
    spec = problem("R2a", "corrected")
    estimate = t3(spec, 4, regime="descriptor", filter="minimax")
    prepared = t2(spec, 4, regime="descriptor", filter="minimax")
    assert estimate.tau is None
    assert estimate.degree == prepared.filter.degree


def test_t3_costs_the_published_encoding_from_its_own_circuit() -> None:
    r2a = problem("R2a", "corrected")
    structured, published = t3(r2a, 3), t3(r2a, 3, encoder="published")
    stacked, _ = encode_standard_linear(r2a, 3, encoder="published")
    assert published.alpha_r == stacked.alpha
    assert published.gap == structured.gap
    assert published.degree > structured.degree
    fitted = t3(r2a, 4, sample_n=(2, 3), encoder="published")
    assert fitted.alpha_r > t3(r2a, 4, sample_n=(2, 3)).alpha_r


@pytest.mark.parametrize("regime", ["standard", "descriptor"])
def test_the_ideal_degree_takes_the_encodings_looseness_out(regime: str) -> None:
    """At alpha = ||R|| the minimax degree falls by about alpha_R / ||R||."""
    result = t3(problem("R2a", "corrected"), 4, regime=regime, filter="minimax")  # type: ignore[arg-type]
    looseness = result.alpha_r / result.norm_r
    assert looseness > 1.0
    assert result.ideal_degree < result.degree
    assert result.degree / result.ideal_degree == pytest.approx(looseness, rel=0.05)


def test_the_fitted_ideal_degree_keeps_the_largest_samples_looseness() -> None:
    direct = t3(problem("R2a", "corrected"), 5, regime="descriptor", filter="minimax")
    fitted = t3(
        problem("R2a", "corrected"), 5, sample_n=(3, 4, 5), regime="descriptor", filter="minimax"
    )
    assert fitted.alpha_r / fitted.norm_r == pytest.approx(direct.alpha_r / direct.norm_r)
