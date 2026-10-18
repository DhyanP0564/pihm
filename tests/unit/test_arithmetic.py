"""Exhaustive checks of the Cuccaro adder and the register subtractor."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from pihm.circuits.arithmetic import register_subtractor, ripple_carry_adder


def _basis_index(carry_in: int, a: int, b: int, extra: int, n: int) -> int:
    """Little-endian index for (carry_in, a, b, extra) on the adder's qubit layout."""
    return (extra << (2 * n + 1)) | (b << (n + 1)) | (a << 1) | carry_in


def _evolve_and_decode(
    circuit: QuantumCircuit, carry_in: int, a: int, b: int, extra: int, n: int
) -> tuple[int, int, int, int]:
    """Return (carry_out, a_out, b_out, extra_out) after evolving one basis state."""
    idx = _basis_index(carry_in, a, b, extra, n)
    vector = np.zeros(1 << (2 * n + 2), dtype=complex)
    vector[idx] = 1.0
    out = Statevector(vector).evolve(circuit)
    (nonzero,) = np.nonzero(np.abs(out.data) > 1e-9)
    assert nonzero.size == 1
    out_idx = int(nonzero[0])
    N = 1 << n
    return (
        out_idx & 1,
        (out_idx >> 1) & (N - 1),
        (out_idx >> (n + 1)) & (N - 1),
        (out_idx >> (2 * n + 1)) & 1,
    )


@pytest.mark.parametrize("n", range(1, 5))
def test_adder_is_exact_addition(n: int) -> None:
    """B <- (a+b) mod N, overflow <- true carry, a and carry_in restored: every (a, b) pair."""
    circuit = ripple_carry_adder(n)
    N = 1 << n
    for a in range(N):
        for b in range(N):
            carry_out, a_out, b_out, overflow = _evolve_and_decode(circuit, 0, a, b, 0, n)
            s = a + b
            assert carry_out == 0
            assert a_out == a
            assert b_out == s % N
            assert overflow == (1 if s >= N else 0)


@pytest.mark.parametrize("n", range(1, 5))
def test_subtractor_is_exact_subtraction_with_borrow(n: int) -> None:
    """B <- (b-a) mod N, borrow <- [b < a], a and carry_in restored: every (a, b) pair."""
    circuit = register_subtractor(n)
    N = 1 << n
    for a in range(N):
        for b in range(N):
            carry_out, a_out, b_out, borrow = _evolve_and_decode(circuit, 0, a, b, 0, n)
            assert carry_out == 0
            assert a_out == a
            assert b_out == (b - a) % N
            assert borrow == (1 if b < a else 0)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_subtractor_is_the_adders_inverse(n: int) -> None:
    """register_subtractor is built as ripple_carry_adder(n).inverse(); check they cancel."""
    add = ripple_carry_adder(n)
    sub = register_subtractor(n)
    combined = Operator(sub).data @ Operator(add).data
    np.testing.assert_allclose(combined, np.eye(1 << (2 * n + 2)), atol=1e-10)
