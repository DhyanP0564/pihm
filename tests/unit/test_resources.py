"""Checks of resource counting and power-law fits."""

from __future__ import annotations

import pytest

from pihm.circuits.folds import fold_encoding
from pihm.circuits.standard import structured_derivative_encoding
from pihm.resources import count_resources, fit_power_law


def test_count_resources_reports_a_consistent_register_split() -> None:
    n = 2
    E = structured_derivative_encoding(n)
    counts = count_resources(E)
    assert counts.n_sys_qubits == E.n_sys_qubits
    assert counts.n_anc_qubits == E.n_anc_qubits
    assert counts.n_qubits == E.n_sys_qubits + E.n_anc_qubits
    assert counts.alpha == E.alpha
    assert counts.cx_count > 0
    assert counts.depth > 0
    assert counts.transpiled_gate_counts.get("cx", 0) == counts.cx_count
    assert sum(counts.logical_gate_counts.values()) > 0


def test_fit_power_law_recovers_a_known_exponent() -> None:
    ns = [2.0, 4.0, 8.0, 16.0, 32.0]
    coefficient, exponent = fit_power_law(ns, [3.0 * n**1.5 for n in ns])
    assert coefficient == pytest.approx(3.0, rel=1e-6)
    assert exponent == pytest.approx(1.5, rel=1e-6)


def test_fit_power_law_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_power_law([1.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        fit_power_law([1.0, -2.0], [1.0, 2.0])


def test_dense_gates_are_counted_apart_as_cited_primitives() -> None:
    """The fold's three Chebyshev–Gauss transforms are cited, not synthesised; G has none."""
    assert count_resources(fold_encoding(3)).cited_count == 3
    assert count_resources(structured_derivative_encoding(3)).cited_count == 0
