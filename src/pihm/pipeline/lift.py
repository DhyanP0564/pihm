r"""The ``"lift"`` stage: a Carleman lift's coupling alone, on a circuit, in either layout.

A run encodes the coupling :math:`\mathbb{A}_K` of a Fourier–Galerkin problem, R9 or R10, with its
lift's level scale, in one of two layouts:

``"symmetric"``
    the pipeline's own: one component per monomial, :math:`\mathbb{A}_K` on the field register as
    a sum of matchings, padded to a
    power of two;
``"tensor"``
    level :math:`k` as the whole tensor :math:`v^{\otimes k}` in :math:`k` Fourier slots,
    :math:`2K - 1` structured terms with :math:`F_2` the fold of the equation's symbols.

The coupling has no time axis and no form, so a lift record's ``n`` is ``(0,)`` and its regime
the default. Its metrics (:func:`lift_metrics`) are the layout's size (modes, components, terms,
qubits), :math:`\alpha` against :math:`\|\mathbb{A}_K\|` in the layout's own coordinates (the
two layouts' norms are of one operator in two coordinates, not equal), the verification and the
gate counts. The verification is ``"exact"`` (the circuit's block on the components against
:math:`\mathbb{A}_K`, and the tensor layout's leak out of its physical states), ``"ir"`` (the
terms reconstructed without the circuit: the matchings summed, or the tensor terms applied on
register values, :func:`pihm.circuits.tensor.tensor_ir`) or ``"none"``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import svds

from pihm.assemble.systems import coupling, lift_scale
from pihm.carleman import tensor_terms
from pihm.circuits.lcu import compile_system
from pihm.circuits.primitives import BlockEncoding, top_left_block
from pihm.circuits.tensor import (
    TensorLayout,
    physical_block,
    tensor_encoding,
    tensor_ir,
    tensor_target,
)
from pihm.ir import SystemIR, matching_terms
from pihm.pipeline.records import JSONValue
from pihm.pipeline.spec import RunSpec
from pihm.problems import ladder
from pihm.resources import count_resources

__all__ = ["LIFT_TOLERANCE", "NORM_LIMIT", "lift_metrics", "run_lift"]

#: The verification's bar on the largest entry's error, relative to the coupling's largest entry.
LIFT_TOLERANCE: Final = 1e-10

#: The largest number of components whose :math:`\|\mathbb{A}_K\|` a run computes.
NORM_LIMIT: Final = 2_000_000


def _largest(matrix: sp.csr_array) -> float:
    """Return a sparse matrix's largest singular value, or NaN past :data:`NORM_LIMIT`."""
    if matrix.shape[0] > NORM_LIMIT:
        return float("nan")
    if min(matrix.shape) < 200:
        return float(np.linalg.norm(matrix.toarray(), 2))
    return float(svds(matrix, k=1, return_singular_vectors=False, tol=1e-6)[0])


def _relative(error: float, target: sp.csr_array) -> float:
    """Return an error relative to the target's largest entry (at least one)."""
    return error / max(1.0, float(abs(target).max()))


def _symmetric(
    spec: RunSpec, order: int, timings: dict[str, float]
) -> tuple[BlockEncoding, dict[str, JSONValue]]:
    """Encode and check the symmetric layout's coupling, as matchings on the field register."""
    started = time.perf_counter()
    A = coupling(spec.problem, order)
    blocks = 1 << (A.shape[0] - 1).bit_length()
    qubits = (blocks - 1).bit_length()
    entries = sp.coo_array(A)
    padded = sp.csr_array(
        sp.coo_array((entries.data, (entries.row, entries.col)), shape=(blocks, blocks))
    )
    ir = SystemIR(tuple(matching_terms(padded.toarray())), qubits, (), ())
    encoding = compile_system(ir, label="A_K")
    timings["encode"] = time.perf_counter() - started
    started = time.perf_counter()
    error: float | None = None
    if spec.verify == "exact":
        error = float(np.max(np.abs(top_left_block(encoding) - padded.toarray())))
    elif spec.verify == "ir":
        error = float(abs(ir.to_sparse() - padded).max())
    timings["verify"] = time.perf_counter() - started
    metrics: dict[str, JSONValue] = {
        "components": int(A.shape[0]),
        "terms": len(ir.terms),
        "norm_A": _largest(sp.csr_array(A)),
        "verify_max_error": None if error is None else _relative(error, padded),
        "leak": None,
    }
    return encoding, metrics


def _tensor(
    spec: RunSpec, order: int, timings: dict[str, float]
) -> tuple[BlockEncoding, dict[str, JSONValue]]:
    """Encode and check the tensor layout's coupling."""
    started = time.perf_counter()
    galerkin = ladder.galerkin(spec.problem.name)
    scale = lift_scale(spec.problem, order)
    layout = TensorLayout.of(galerkin, order)
    encoding = tensor_encoding(galerkin, order, scale)
    timings["encode"] = time.perf_counter() - started
    components = int(layout.physical().size)
    target = tensor_target(galerkin, order, scale) if components <= NORM_LIMIT else None
    started = time.perf_counter()
    error: float | None = None
    leak: float | None = None
    if spec.verify == "exact" and target is not None:
        block, leak = physical_block(encoding, layout)
        error = _relative(float(np.max(np.abs(block - target.toarray()))), target)
    elif spec.verify == "ir" and target is not None:
        error = _relative(float(abs(tensor_ir(galerkin, order, scale) - target).max()), target)
    timings["verify"] = time.perf_counter() - started
    metrics: dict[str, JSONValue] = {
        "components": components,
        "terms": len(tensor_terms(order)),
        "norm_A": float("nan") if target is None else _largest(target),
        "verify_max_error": error,
        "leak": leak,
    }
    return encoding, metrics


def lift_metrics(
    spec: RunSpec, encoding: BlockEncoding, checked: Mapping[str, JSONValue], resources: bool
) -> dict[str, JSONValue]:
    """Return a ``"lift"`` run's scalar metrics (module documentation)."""
    problem = spec.problem
    error = checked["verify_max_error"]
    norm = checked["norm_A"]
    assert isinstance(norm, float)
    counts = count_resources(encoding) if resources else None
    return {
        "problem": problem.name,
        "variant": problem.variant,
        "rung": ladder.rung(problem.name),
        "label": spec.label,
        "lift": spec.lift,
        "carleman": spec.carleman or 0,
        "modes": len(problem.fields),
        **checked,
        "alpha": encoding.alpha,
        "alpha_over_norm": encoding.alpha / norm if np.isfinite(norm) and norm > 0 else None,
        "n_sys_qubits": encoding.n_sys_qubits,
        "n_anc_qubits": encoding.n_anc_qubits,
        "n_qubits_total": encoding.n_qubits,
        "verify_mode": spec.verify,
        "verify_passed": None if error is None else bool(float(error) <= LIFT_TOLERANCE),
        "verify_tolerance": LIFT_TOLERANCE,
        "resources_measured": counts is not None,
        "cx_count": counts.cx_count if counts else None,
        "one_qubit_count": counts.one_qubit_count if counts else None,
        "depth": counts.depth if counts else None,
    }


def run_lift(
    spec: RunSpec, timings: dict[str, float]
) -> tuple[Mapping[str, JSONValue], dict[str, NDArray[Any]]]:
    """Run the ``"lift"`` stage: encode the coupling in the run's layout, check it, count it."""
    if spec.carleman is None:
        raise ValueError("a lift run needs a Carleman order")
    build = _tensor if spec.lift == "tensor" else _symmetric
    encoding, checked = build(spec, spec.carleman, timings)
    started = time.perf_counter()
    metrics = lift_metrics(spec, encoding, checked, spec.resources)
    timings["resources"] = time.perf_counter() - started
    return metrics, {}
