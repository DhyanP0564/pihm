"""Source injection as circuits, M_r D0(x_s), against the operators."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from pihm import basis
from pihm.circuits.compose import real_state_preparation
from pihm.circuits.primitives import native_circuit, top_left_block
from pihm.circuits.sources import source_encoding
from pihm.problems.functions import polynomial_coefficients, source_coefficients
from pihm.problems.ladder import problem
from pihm.problems.reference import regular_value
from pihm.problems.spec import Term


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("m", [1, 2, 3])
def test_real_state_preparation_prepares_any_real_vector(m: int, seed: int) -> None:
    """Signs and zero subtrees included, |0> goes to the unit vector along the amplitudes."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=1 << m)
    values[rng.uniform(size=values.size) < 0.4] = 0.0
    values[0] = values[0] or -1.0
    circuit = native_circuit(real_state_preparation(values))
    state = Statevector.from_label("0" * m).evolve(circuit).data
    np.testing.assert_allclose(state, values / np.linalg.norm(values), atol=1e-14)


def test_real_state_preparation_refuses_a_bad_length_or_zero() -> None:
    """The register is whole qubits, and a zero vector has no direction."""
    with pytest.raises(ValueError, match="power-of-two"):
        real_state_preparation([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="nonzero"):
        real_state_preparation([0.0, 0.0])


def _source_term(name: str, variant: str) -> Term:
    (equation,) = problem(name, variant).equations
    (term,) = [t for t in equation.terms if not t.factors]
    return term


@pytest.mark.parametrize(
    ("name", "variant"),
    [("R4a", "faithful"), ("R4b", "faithful"), ("R4b", "corrected"), ("R4d", "corrected")],
)
def test_source_encoding_is_the_rank_one_injection(name: str, variant: str) -> None:
    """M_r D0(x_s) = |r̂><tau(x_s)| / f(x_s) on the lifted register, whichever representation."""
    n = 2
    p = problem(name, variant)
    term = _source_term(name, variant)
    encoding = source_encoding(term, p, n)
    (axis,) = p.fields[0].axes
    if term.source is None:
        assert term.multiplier is not None
        r_hat = polynomial_coefficients(term.multiplier, n + 1)
    else:
        r_hat = source_coefficients(term.source, [axis], [n + 1])
    expected = np.outer(r_hat, basis.tau(p.regular.at[0], n)) / regular_value(p)
    assert encoding.shape == expected.shape
    np.testing.assert_allclose(top_left_block(encoding), expected, atol=1e-12)


def test_source_encoding_refuses_a_term_with_factors() -> None:
    """A source term has no factors."""
    (equation,) = problem("R4b", "corrected").equations
    with pytest.raises(ValueError, match="no factors"):
        source_encoding(equation.terms[0], problem("R4b", "corrected"), 2)
