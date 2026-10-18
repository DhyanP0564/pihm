r"""A case's records across the campaigns: its fields, its matrices and its metrics.

A solution viewer shows one case at a time: a problem, a resolution
:math:`n` and a Carleman order :math:`K` (:class:`CaseKey`). :class:`Catalogue`
reads every campaign's summary (``examples/campaigns.yaml``) and :meth:`Catalogue.case` finds each
stage's record of a case in each form, choosing among its records as the reports do:

* the build: the reference build, unlabelled, untruncated and lifted
  (:func:`~pihm.pipeline.results.reference_builds`);
* T2: the primary preparation (:func:`~pihm.pipeline.results.primary_preparations`) from its
  best initial-state candidate (:func:`~pihm.pipeline.summary.keep_best_candidate`);
* T1: the kept candidate of the primary circuit runs, or else its refusal;
* T3: the estimate under the pipeline's filter and encoder (minimax, structured) where one was
  made, otherwise the first;
* the encoding: its most thorough verification;
* the readout: T1's, otherwise T2's;
* the coupling's circuit (the ``lift`` stage, which has no time axis and so no :math:`n`): one
  record per layout at the case's problem and order.

:func:`solution` reads the fields of the case's build and preparation records into a
:class:`Solution`, whose :data:`Family` says how it is drawn. :func:`structure` rebuilds a build's
residual from its specification (:meth:`~pihm.pipeline.spec.RunSpec.from_dict`) for the spy plots
of :math:`R` and :math:`H`, with its blocks marked: the one computation the viewer makes, since
:func:`~pihm.assemble.assemble` is deterministic and fast. :func:`metrics` and :func:`coupling`
set the case's metrics from every stage beside each other.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm.assemble import BlockKind, Regime, Residual, assemble
from pihm.pipeline.catalogue import run_key
from pihm.pipeline.records import loads_strict
from pihm.pipeline.results import NOT_PRIMARY, Results, primary_preparations
from pihm.pipeline.spec import RunSpec
from pihm.pipeline.summary import encoder_of, filter_of, keep_best_candidate
from pihm.problems import ladder
from pihm.problems.periodic import PeriodicSpace, periodic_space
from pihm.problems.spec import Problem, Variant

__all__ = [
    "HAMILTONIAN_LIMIT",
    "REGIMES",
    "Band",
    "CaseKey",
    "CaseRecords",
    "Catalogue",
    "Family",
    "FieldSource",
    "Found",
    "Layer",
    "MetricsTable",
    "Solution",
    "Structure",
    "coupling",
    "family",
    "metrics",
    "solution",
    "structure",
]

#: How a case's fields are drawn, as one field on one axis, several fields on one axis, one field
#: on two axes, or a Fourier–Galerkin state on its periodic domain (Burgers, the vorticity).
Family: TypeAlias = Literal["curve", "fields", "surface", "burgers", "vorticity"]

#: The state a field is decoded from, the classical ground state or a prepared state (T2's
#: exact emulation, T1's simulated circuit).
FieldSource: TypeAlias = Literal["ground", "T2", "T1"]

#: The forms, in the order every table and figure sets them.
REGIMES: Final[tuple[Regime, ...]] = ("standard", "descriptor")

#: The nonzeros of :math:`R` past which :func:`structure` does not form :math:`H = R^\dagger R`.
HAMILTONIAN_LIMIT: Final = 2_000_000

#: The stages the viewer reads; the others (``protocol``, ``filters``) study a choice, not a case.
_STAGES: Final = frozenset({"build", "encode", "stateprep", "readout", "lift"})

#: How thoroughly an encoding was verified, most first.
_VERIFY_RANK: Final[Mapping[str, int]] = {"exact": 3, "probe": 2, "ir": 1, "none": 0}

#: The plural of each row block's kind, for a band of several blocks.
_PLURAL: Final[Mapping[BlockKind, str]] = {
    "equation": "equations",
    "recurrence": "recurrences",
    "constraint": "constraints",
    "padding": "padding blocks",
}

#: The primes of a derivative on one axis.
_PRIMES: Final[Mapping[int, str]] = {1: "′", 2: "″", 3: "‴"}

#: Superscript digits, for a derivative's order on one of several axes.
_SUPERSCRIPTS: Final = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass(frozen=True)
class CaseKey:
    """A case as the viewer chooses it.

    Attributes
    ----------
    problem
        The problem's name, such as ``"R10c-M12"``.
    n
        The resolution, the largest qubit count on an axis (0 for the coupling's circuit alone).
    carleman
        The Carleman lift's order :math:`K`, or ``None`` without a lift.
    variant
        ``"faithful"`` for a paper panel as printed, which only the reproduction runs; the
        problem itself by default.
    """

    problem: str
    n: int
    carleman: int | None = None
    variant: Variant = "corrected"

    def __str__(self) -> str:
        """Return the case as the viewer names it, such as ``"R8, n = 4, K = 3"``."""
        printed = " (faithful)" if self.variant == "faithful" else ""
        order = "" if self.carleman is None else f", K = {self.carleman}"
        return f"{self.problem}{printed}, n = {self.n}{order}"

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> CaseKey:
        """Return the case of a summary row; a build's order 0 and an absent one are no lift."""
        carleman = row.get("carleman")
        order = None if carleman in ("", None, 0) else int(carleman)
        variant: Variant = "faithful" if row.get("variant") == "faithful" else "corrected"
        return cls(str(row["problem"]), int(row["n"]), order, variant)

    @property
    def order(self) -> tuple[int, int, int, int]:
        """The case's place in lists, by the ladder's order, then the variant, n and K."""
        names = ladder.names()
        position = names.index(self.problem) if self.problem in names else len(names)
        return position, self.variant != "faithful", self.n, self.carleman or 0


@dataclass(frozen=True)
class Found:
    """One record: its campaign, its summary row and its directory.

    Attributes
    ----------
    campaign
        The campaign's name in ``examples/campaigns.yaml``.
    row
        Its row of the campaign's ``summary.csv``.
    directory
        The record's directory.
    """

    campaign: str
    row: Mapping[str, Any]
    directory: Path

    @property
    def stage(self) -> str:
        """The record's stage, the first segment of its directory."""
        return str(self.row["directory"]).partition("/")[0]

    @property
    def ok(self) -> bool:
        """Whether the run finished."""
        return self.row.get("status") == "ok"

    @property
    def has_arrays(self) -> bool:
        """Whether the record kept arrays (a state preparation that keeps its field)."""
        return (self.directory / "arrays.npz").exists()

    def arrays(self) -> dict[str, NDArray[Any]]:
        """Return the record's arrays, none if it kept none."""
        if not self.has_arrays:
            return {}
        with np.load(self.directory / "arrays.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def spec(self) -> RunSpec:
        """Return the record's run specification."""
        return RunSpec.from_dict(self._spec())

    def key(self, labels: Mapping[str, str]) -> str:
        """Return the record's run key (:func:`~pihm.pipeline.catalogue.run_key`)."""
        return run_key(self._spec(), labels)

    def _spec(self) -> Mapping[str, Any]:
        """Return the specification its ``meta.json`` holds."""
        meta = loads_strict((self.directory / "meta.json").read_text())
        spec: Mapping[str, Any] = meta["spec"]
        return spec


@dataclass(frozen=True)
class CaseRecords:
    """A case's chosen record of each stage in each form (module documentation).

    Each mapping holds a form's record, by form, where the campaigns have one; ``lift`` holds
    the coupling's circuit by layout.
    """

    key: CaseKey
    build: Mapping[str, Found]
    t2: Mapping[str, Found]
    t1: Mapping[str, Found]
    t3: Mapping[str, Found]
    encode: Mapping[str, Found]
    readout: Mapping[str, Found]
    lift: Mapping[str, Found]

    def stages(self) -> dict[str, Mapping[str, Found]]:
        """Return every stage's records, named as the viewer lists them."""
        return {
            "build": self.build,
            "encoding": self.encode,
            "T2": self.t2,
            "T1": self.t1,
            "T3": self.t3,
            "readout": self.readout,
            "coupling": self.lift,
        }

    def run_keys(self, labels: Mapping[str, str]) -> list[tuple[str, str, str]]:
        """Return each record's stage, form (or layout) and run key, as a reader quotes it."""
        return [
            (stage, name, found.key(labels))
            for stage, chosen in self.stages().items()
            for name, found in chosen.items()
        ]


@dataclass(frozen=True)
class Catalogue:
    """Every record the viewer reads, by case.

    Attributes
    ----------
    results
        The campaigns' results directories.
    records
        Every record of each case; the coupling's circuit is filed under :math:`n = 0`.
    """

    results: Results
    records: Mapping[CaseKey, tuple[Found, ...]]

    @classmethod
    def read(cls, results: Results, campaigns: Iterable[str] | None = None) -> Catalogue:
        """Read ``campaigns`` (by default every one), skipping any without a summary yet."""
        found: dict[CaseKey, list[Found]] = defaultdict(list)
        for campaign in campaigns if campaigns is not None else results.directories:
            if results.missing([campaign]):
                continue
            root = results.directory(campaign)
            for row in results.rows(campaign):
                record = Found(campaign, row, root / str(row["directory"]))
                if record.stage in _STAGES:
                    found[CaseKey.of(row)].append(record)
        return cls(results, {key: tuple(records) for key, records in found.items()})

    def cases(self, problem: str | None = None) -> list[CaseKey]:
        """Return the cases, of one problem or of all, in the ladder's order."""
        keys = [key for key in self.records if problem is None or key.problem == problem]
        return sorted(keys, key=lambda key: key.order)

    def problems(self) -> list[str]:
        """Return the problems that have a case, in the ladder's order."""
        return list(dict.fromkeys(key.problem for key in self.cases()))

    def case(self, key: CaseKey) -> CaseRecords:
        """Return the case's chosen record of each stage in each form (module documentation)."""
        found = self.records.get(key, ())
        circuit = CaseKey(key.problem, 0, key.carleman, key.variant)
        couplings = [f for f in self.records.get(circuit, ()) if f.stage == "lift"]
        return CaseRecords(
            key=key,
            build=_by_regime(found, "build", _reference_build),
            t2=_by_regime(found, "stateprep", _primary_t2),
            t1=_by_regime(found, "stateprep", _kept_t1),
            t3=_by_regime(found, "stateprep", _estimate),
            encode=_by_regime(found, "encode", _encoding),
            readout=_by_regime(found, "readout", _readout),
            lift={
                str(f.row["lift"]): f
                for f in sorted(couplings, key=lambda f: str(f.row.get("lift")))
                if f.ok
            },
        )

    def richest(self, problem: str) -> CaseKey:
        """Return the problem's case with the most fields to draw, at the largest n among them.

        The most forms with a build, then the most with a T2 preparation (whose record
        keeps its field), then the highest Carleman order and the largest n; T1's few small
        runs do not decide it.

        Raises
        ------
        KeyError
            If the problem has no case.
        """
        keys = self.cases(problem)
        if not keys:
            raise KeyError(f"{problem} has no case in the campaigns read")

        def score(key: CaseKey) -> tuple[int, int, int, int]:
            records = self.case(key)
            return len(records.build), len(records.t2), key.carleman or 0, key.n

        return max(keys, key=score)

    def coverage(self) -> list[dict[str, Any]]:
        """Return, per family, its cases, those with a field, and the preparations' fields.

        ``prepared`` counts the chosen T1 and T2 records that keep their field, ``earlier``
        those whose records kept none.
        """
        table: dict[Family, dict[str, Any]] = {}
        for key in self.cases():
            if key.n == 0:
                continue
            records = self.case(key)
            chosen = [f for f in (*records.t2.values(), *records.t1.values()) if f.ok]
            entry = table.setdefault(
                family(ladder.problem(key.problem, key.variant)),
                {"problems": set(), "cases": 0, "with_field": 0, "prepared": 0, "earlier": 0},
            )
            entry["problems"].add(key.problem)
            entry["cases"] += 1
            entry["with_field"] += bool(records.build) or any(f.has_arrays for f in chosen)
            entry["prepared"] += sum(f.has_arrays for f in chosen)
            entry["earlier"] += sum(not f.has_arrays for f in chosen)
        order: tuple[Family, ...] = ("curve", "fields", "surface", "burgers", "vorticity")
        return [
            {"family": name, **table[name], "problems": len(table[name]["problems"])}
            for name in order
            if name in table
        ]


def _by_regime(
    found: Sequence[Found], stage: str, choose: Callable[[list[Found]], Found | None]
) -> dict[str, Found]:
    """Return, in each form that has one, the record ``choose`` picks of the stage's records."""
    chosen: dict[str, Found] = {}
    for regime in REGIMES:
        candidate = choose([f for f in found if f.stage == stage and f.row["regime"] == regime])
        if candidate is not None:
            chosen[regime] = candidate
    return chosen


def _back(found: Sequence[Found], rows: Iterable[Mapping[str, Any]]) -> list[Found]:
    """Return the records of ``rows``, which a selection returned as copies of their rows."""
    records = {str(f.row["directory"]): f for f in found}
    return [records[str(row["directory"])] for row in rows]


def _best(found: Sequence[Found], rank: Callable[[Found], tuple[Any, ...]]) -> Found | None:
    """Return the record of largest ``rank``, the first on a tie; ``None`` if there is none."""
    if not found:
        return None
    return max(reversed(found), key=rank)


def _reference_build(found: list[Found]) -> Found | None:
    """Return the reference build: ``ok``, unlabelled, untruncated and lifted."""
    return next(
        (
            f
            for f in found
            if f.ok
            and f.row.get("label") == ""
            and f.row.get("truncation") == "none"
            and f.row.get("mult") == "lift"
        ),
        None,
    )


def _primary_t2(found: list[Found]) -> Found | None:
    """Return the primary T2 preparation, its best candidate, the minimax filter's first."""
    kept = keep_best_candidate(primary_preparations(f.row for f in found))
    return _best(_back(found, kept), lambda f: (filter_of(f.row) == "minimax",))


def _kept_t1(found: list[Found]) -> Found | None:
    """Return the kept primary circuit run, the minimax filter's first, else a refusal."""
    rows = [
        f.row for f in found if f.row.get("tier") == "T1" and f.row.get("label") not in NOT_PRIMARY
    ]
    return _best(
        _back(found, keep_best_candidate(rows)),
        lambda f: (f.ok, filter_of(f.row) == "minimax"),
    )


def _estimate(found: list[Found]) -> Found | None:
    """Return the estimate under the pipeline's filter and encoder where made, else the first."""
    estimates = [f for f in found if f.row.get("tier") == "T3" and f.row.get("label") == ""]
    return _best(
        estimates,
        lambda f: (
            f.row.get("status") in ("ok", "infeasible"),
            filter_of(f.row) == "minimax",
            encoder_of(f.row) == "structured",
        ),
    )


def _encoding(found: list[Found]) -> Found | None:
    """Return the encoding verified most thoroughly, with its gates counted where they were."""
    return _best(
        found,
        lambda f: (
            f.ok,
            f.row.get("verify_passed") is True,
            _VERIFY_RANK.get(str(f.row.get("verify_mode")), 0),
            f.row.get("resources_measured") is True,
        ),
    )


def _readout(found: list[Found]) -> Found | None:
    """Return the readout by the simulated circuit (T1) where it ran, else the exact one."""
    return _best(found, lambda f: (f.ok, f.row.get("tier") == "T1"))


# ---------------------------------------------------------------------------- the fields


def family(problem: Problem) -> Family:
    """Return how a problem's fields are drawn (:data:`Family`)."""
    space = periodic_space(problem.name)
    if space is not None:
        return space.equation
    if len(problem.fields) > 1:
        return "fields"
    return "surface" if len(problem.fields[0].axes) == 2 else "curve"


@dataclass(frozen=True, eq=False)
class Layer:
    """One decoded field of a case.

    Attributes
    ----------
    regime
        The form.
    source
        The state it is decoded from.
    field
        The field on the comparison grid, stacked as :attr:`Solution.reference` is.
    record
        The record it was read from.
    """

    regime: Regime
    source: FieldSource
    field: NDArray[np.float64]
    record: Found


@dataclass(frozen=True, eq=False)
class Solution:
    r"""A case's decoded fields in both forms, beside the reference solution.

    Attributes
    ----------
    records
        The case's chosen records.
    problem
        The problem, as its records hold it.
    family
        How it is drawn.
    axes
        The comparison grid's coordinates on the Chebyshev interval, one vector per axis.
    reference
        The reference solution on the grid: a system's fields (a Fourier–Galerkin state's modes)
        along a first axis.
    layers
        The fields decoded from the ground state and the prepared states, standard form first.
    space
        The periodic domain of a Fourier–Galerkin problem (R9, R10), ``None`` for any other.
    notes
        What is missing and why, such as a preparation made before its field was kept.
    """

    records: CaseRecords
    problem: Problem
    family: Family
    axes: tuple[NDArray[np.float64], ...]
    reference: NDArray[np.float64]
    layers: tuple[Layer, ...]
    space: PeriodicSpace | None
    notes: tuple[str, ...]

    @property
    def key(self) -> CaseKey:
        """The case."""
        return self.records.key

    @property
    def axis_names(self) -> tuple[str, ...]:
        """The names of the field's axes, in the grid's order."""
        return self.problem.fields[0].axes

    @property
    def time_axis(self) -> int | None:
        """The grid axis that is time (named ``t``), or ``None`` for a stationary problem."""
        return self.axis_names.index("t") if "t" in self.axis_names else None

    @property
    def seconds(self) -> NDArray[np.float64]:
        """A Fourier–Galerkin case's physical times :math:`s` on the grid's time axis.

        Raises
        ------
        ValueError
            For a case without a periodic domain.
        """
        if self.space is None:
            raise ValueError("only a Fourier-Galerkin case has a physical time")
        return self.space.seconds(self.axes[0])


def solution(records: CaseRecords, problem: Problem | None = None) -> Solution | None:
    """Return the case's decoded fields, or ``None`` if no record of it holds a field.

    Parameters
    ----------
    records
        The case's chosen records (:meth:`Catalogue.case`).
    problem
        The problem, by default read from the first record that has one.
    """
    layers: list[Layer] = []
    notes: list[str] = []
    grid: dict[str, NDArray[Any]] = {}
    for regime in REGIMES:
        build = records.build.get(regime)
        if build is not None:
            arrays = build.arrays()
            grid = grid or arrays
            layers.append(Layer(regime, "ground", np.real(arrays["field"]), build))
        prepared: tuple[tuple[FieldSource, Mapping[str, Found]], ...] = (
            ("T2", records.t2),
            ("T1", records.t1),
        )
        for source, chosen in prepared:
            found = chosen.get(regime)
            if found is None or not found.ok:
                continue
            arrays = found.arrays()
            if "field" not in arrays:
                notes.append(
                    f"The {regime} form's {source} record kept no field; its metrics stand alone."
                )
                continue
            grid = grid or arrays
            layers.append(Layer(regime, source, np.real(arrays["field"]), found))
    if not grid:
        return None
    if problem is None:
        problem = layers[0].record.spec().problem
    count = sum(key.startswith("grid_") for key in grid)
    return Solution(
        records=records,
        problem=problem,
        family=family(problem),
        axes=tuple(np.asarray(grid[f"grid_{index}"]) for index in range(count)),
        reference=np.real(grid["reference"]),
        layers=tuple(layers),
        space=periodic_space(problem.name),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------- the matrices


@dataclass(frozen=True)
class Band:
    """A labelled run of rows or columns of a matrix, from ``start`` up to ``stop``."""

    label: str
    start: int
    stop: int


@dataclass(frozen=True, eq=False)
class Structure:
    r"""A build's stacked residual and Hamiltonian, with their blocks marked, for a spy plot.

    Attributes
    ----------
    regime
        The form.
    residual
        :math:`R`, its rows that are identically zero dropped, as
        :meth:`~pihm.assemble.Residual.matrix` gives it.
    hamiltonian
        :math:`H = R^\dagger R`, or ``None`` where :math:`R` has more than
        :data:`HAMILTONIAN_LIMIT` nonzeros.
    rows
        :math:`R`'s row blocks, consecutive blocks of one kind together: the equation, the
        descriptor's recurrences, the constraints, the padding.
    columns
        The state's blocks, those carrying the same thing together: the field and its
        derivatives, a system's fields, a Carleman lift's degrees, the padding.
    """

    regime: Regime
    residual: sp.csr_array
    hamiltonian: sp.csr_array | None
    rows: tuple[Band, ...]
    columns: tuple[Band, ...]


def structure(record: Found) -> Structure:
    """Rebuild a build record's residual from its specification, and mark its blocks."""
    spec = record.spec()
    residual = assemble(
        spec.problem,
        spec.n,
        regime=spec.regime,
        mult=spec.mult,
        truncation=spec.truncation,
        rescale=spec.rescale,
        carleman=spec.carleman,
        layout=spec.layout,
    )
    pieces: list[sp.csr_array] = []
    rows: list[Band] = []
    for kind, group in itertools.groupby(residual.blocks, key=lambda block: block.kind):
        blocks = [_nonzero_rows(block.matrix) for block in group]
        size = sum(int(block.shape[0]) for block in blocks)
        start = rows[-1].stop if rows else 0
        if size:
            label = kind if len(blocks) == 1 else f"{len(blocks)} {_PLURAL[kind]}"
            rows.append(Band(label, start, start + size))
        pieces.extend(blocks)
    R = sp.csr_array(sp.vstack(pieces, format="csr"))
    H = sp.csr_array(R.conj().T @ R) if R.nnz <= HAMILTONIAN_LIMIT else None
    return Structure(spec.regime, R, H, tuple(rows), _columns(residual))


def _nonzero_rows(matrix: sp.csr_array) -> sp.csr_array:
    """Return the block without its identically zero rows."""
    block = sp.csr_array(matrix, copy=True)
    block.eliminate_zeros()
    return sp.csr_array(block[np.flatnonzero(np.diff(block.indptr))])


def _columns(residual: Residual) -> tuple[Band, ...]:
    """Return the state's blocks, each run of blocks with one label as one band."""
    size = residual.dimension // len(residual.derivatives)
    labels = [_column_label(residual, index) for index in range(len(residual.derivatives))]
    bands: list[Band] = []
    for label, group in itertools.groupby(enumerate(labels), key=lambda item: item[1]):
        indices = [index for index, _ in group]
        count = f" ×{len(indices)}" if len(indices) > 1 else ""
        bands.append(Band(label + count, indices[0] * size, (indices[-1] + 1) * size))
    return tuple(bands)


def _column_label(residual: Residual, index: int) -> str:
    """Return what the state's block ``index`` carries."""
    problem = residual.problem
    derivative = residual.derivatives[index]
    if derivative is None:
        return "padding"
    names = [field.name for field in problem.fields]
    if residual.nonlinearity == "doubled":
        return f"{names[0]} ⊗ {names[0]}"
    if residual.monomials:
        monomial = residual.monomials[index]
        assert monomial is not None
        if residual.carleman_order == 1:
            return names[monomial.index(1)]
        return f"degree {sum(monomial)}"
    return _derivative_label(names[0], problem.fields[0].axes, derivative)


def _derivative_label(name: str, axes: Sequence[str], orders: Sequence[int]) -> str:
    """Return a derivative's name: ``f″`` on one axis, ``∂t f`` or ``∂x² f`` on several."""
    if not any(orders):
        return name
    if len(orders) == 1:
        return name + _PRIMES.get(orders[0], f"^({orders[0]})")
    parts = "".join(
        f"∂{axis}{str(order).translate(_SUPERSCRIPTS) if order > 1 else ''}"
        for axis, order in zip(axes, orders, strict=True)
        if order
    )
    return f"{parts} {name}"


# ---------------------------------------------------------------------------- the metrics


@dataclass(frozen=True)
class MetricsTable:
    """A case's metrics, one column per form (or per layout, for the coupling's circuit).

    Attributes
    ----------
    columns
        The columns' names.
    rows
        Each metric's section, name and value per column (``None`` where a column has none).
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[str, str, tuple[Any, ...]], ...]


#: A metric: its name, and how it is read from a summary row.
_Metric: TypeAlias = tuple[str, Callable[[Mapping[str, Any]], Any]]


def _column(name: str) -> Callable[[Mapping[str, Any]], Any]:
    """Return the reader of one summary column."""
    return lambda row: row.get(name)


def _qubits(row: Mapping[str, Any]) -> int:
    """Return the state's qubits, from its dimension."""
    return int(row["dimension"]).bit_length() - 1


#: Each section's stage (a :meth:`CaseRecords.stages` name) and metrics, in the order shown.
_SECTIONS: Final[tuple[tuple[str, str, tuple[_Metric, ...]], ...]] = (
    (
        "classical build",
        "build",
        (
            ("state qubits", _qubits),
            ("state dimension", _column("dimension")),
            ("nonzeros of R", _column("nonzeros")),
            ("‖H‖", _column("hamiltonian_norm")),
            ("gap Δ", _column("gap")),
            ("relative gap Δ/‖H‖", _column("relative_gap")),
            ("kernel dimension", _column("kernel_dimension")),
            ("ground state trusted", _column("trusted")),
            ("η_e", _column("eta")),
            ("ideal η_e = ‖b‖²", _column("eta_ideal")),
            ("field error L∞ (ε_disc)", _column("field_err_linf")),
            ("field error, relative L²", _column("field_err_l2")),
            ("the lift's error L∞", _column("lift_err_linf")),
            ("readout amplification", _column("readout_amplification")),
            ("wall time (s)", _column("time_total")),
        ),
    ),
    (
        "encoding (Pillar 1)",
        "encoding",
        (
            ("α_R", _column("alpha_R")),
            ("system qubits", _column("n_sys_qubits")),
            ("ancilla qubits", _column("n_anc_qubits")),
            ("total qubits", _column("n_qubits_total")),
            ("CX gates", _column("cx_count")),
            ("one-qubit gates", _column("one_qubit_count")),
            ("circuit depth", _column("depth")),
            ("verification", _column("verify_mode")),
            ("verification error", _column("verify_max_error")),
        ),
    ),
    (
        "preparation, T2 (exact emulation)",
        "T2",
        (
            ("filter", filter_of),
            ("filter degree", _column("degree")),
            ("success (post-selection) probability", _column("success_probability")),
            ("initial overlap γ²", _column("gamma2")),
            ("infidelity", _column("infidelity")),
            ("infidelity bound", _column("infidelity_bound")),
            ("distance from the ground state", _column("state_deviation")),
            ("field error L∞", _column("field_err_linf")),
            ("field bound", _column("field_bound")),
            ("field within its bound", _column("field_within_bound")),
            ("wall time (s)", _column("time_total")),
        ),
    ),
    (
        "preparation, T1 (circuit simulated)",
        "T1",
        (
            ("status", _column("status")),
            ("qubits", _column("n_qubits")),
            ("filter degree", _column("degree")),
            ("success (post-selection) probability", _column("success_probability")),
            ("infidelity", _column("infidelity")),
            ("agreement with T2", _column("t2_agreement")),
            ("refused because", _column("infeasible_reasons")),
            ("wall time (s)", _column("time_total")),
        ),
    ),
    (
        "estimate, T3",
        "T3",
        (
            ("status", _column("status")),
            ("filter", filter_of),
            ("filter degree", _column("degree")),
            ("qubits", _column("n_qubits")),
            ("gates", _column("gate_count")),
            ("predicted success probability", _column("predicted_p")),
            ("amplitude amplification rounds", _column("amplification_rounds")),
            ("preparations", _column("preparations")),
            ("beyond the simulation ceiling", _column("beyond_ceiling")),
        ),
    ),
    (
        "readout (Eqs 36–39)",
        "readout",
        (
            ("tier", _column("tier")),
            ("prepared-state probability", _column("prepared_probability")),
            ("combined-state probability", _column("combined_probability")),
            ("field error L∞", _column("field_err_linf")),
            ("η_e against T2", _column("eta_agreement")),
        ),
    ),
)

#: The coupling circuit's metrics (the ``lift`` stage, :mod:`pihm.pipeline.lift`).
_COUPLING: Final[tuple[_Metric, ...]] = (
    ("components", _column("components")),
    ("terms", _column("terms")),
    ("α", _column("alpha")),
    ("α/‖𝔸_K‖", _column("alpha_over_norm")),
    ("total qubits", _column("n_qubits_total")),
    ("ancilla qubits", _column("n_anc_qubits")),
    ("CX gates", _column("cx_count")),
    ("circuit depth", _column("depth")),
    ("verification", _column("verify_mode")),
    ("verification error", _column("verify_max_error")),
)


def _present(value: Any) -> bool:
    """Whether a summary cell holds a value (not empty, not NaN)."""
    if value is None or value == "":
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _table(
    columns: Sequence[str],
    sections: Iterable[tuple[str, Mapping[str, Found], Sequence[_Metric]]],
) -> MetricsTable:
    """Return the metrics each column's record has; a row no column has is left out."""
    rows = []
    for title, chosen, metrics_ in sections:
        for name, read in metrics_:
            values = tuple(
                read(chosen[column].row) if column in chosen else None for column in columns
            )
            values = tuple(value if _present(value) else None for value in values)
            if any(value is not None for value in values):
                rows.append((title, name, values))
    return MetricsTable(tuple(columns), tuple(rows))


def metrics(records: CaseRecords) -> MetricsTable:
    """Return the case's metrics from every stage, a column per form.

    The classical build, the encoding, T2, T1, T3 and the readout, each where a form has its
    record (:class:`CaseRecords`); a metric neither form has is left out.
    """
    stages = records.stages()
    return _table(REGIMES, ((title, stages[stage], rows) for title, stage, rows in _SECTIONS))


def coupling(records: CaseRecords) -> MetricsTable | None:
    """Return the Carleman coupling's circuit, a column per layout, or ``None`` if not built."""
    if not records.lift:
        return None
    return _table(tuple(records.lift), [("the coupling's circuit", records.lift, _COUPLING)])
