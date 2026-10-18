"""Collect run records into one ``summary.csv``.

The analysis reads only this table and the records' NPZ arrays. Each row is one
record: its identity (configuration hash, status and directory), its wall time, and every
scalar metric, with lists joined by ``;``. Columns are the union over all records, identity
first and the rest in alphabetical order, so that tables from different phases merge.

Tables join across campaigns by case, a problem, its variant and its resolution (:func:`cases`):
:func:`side_by_side` sets two forms' records of the same cases next to each other, and
:func:`encoding_comparison` does so for Pillar 1, each encoding's looseness against the norm its
build record measured.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pihm.pipeline.records import Record, read_record
from pihm.qsp.initial import GEOMETRIC_R

__all__ = [
    "IDENTITY",
    "Case",
    "candidate_rank",
    "cases",
    "collect",
    "encoder_of",
    "encoding_comparison",
    "filter_of",
    "filter_prose",
    "keep_best_candidate",
    "read_arrays",
    "read_summary",
    "side_by_side",
    "summary_rows",
]

#: The key of a case, its problem, variant and resolution (the largest qubit count per axis).
Case: TypeAlias = tuple[str, str, int]

#: The leading columns of ``summary.csv``, in order.
IDENTITY = (
    "config_hash",
    "status",
    "problem",
    "variant",
    "label",
    "regime",
    "n",
    "directory",
    "time_total",
    "peak_memory_gib",
)


def _cell(value: Any) -> Any:
    """Format one metric for CSV: lists joined by ``;``, non-finite floats by name."""
    if isinstance(value, list):
        return ";".join(str(_cell(item)) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def summary_rows(records: Iterable[tuple[Path, Record]], root: Path) -> list[dict[str, Any]]:
    """Return one flat row per record, with directories relative to ``root``."""
    rows = []
    for directory, record in records:
        row: dict[str, Any] = {key: _cell(value) for key, value in record.metrics.items()}
        spec = record.spec
        problem = spec.get("problem", {})
        row.update(
            config_hash=record.config_hash,
            status=record.status,
            problem=row.get("problem", problem.get("name", "")),
            variant=row.get("variant", problem.get("variant", "corrected")),
            label=spec.get("label", ""),
            regime=spec.get("regime", ""),
            n=row.get("n", max(spec.get("n", [0]))),
            directory=str(directory.relative_to(root)),
            time_total=record.timings.get("total", float("nan")),
            peak_memory_gib=(
                float("nan") if record.peak_memory_gib is None else record.peak_memory_gib
            ),
        )
        rows.append(row)
    return rows


def collect(root: Path, output: Path | None = None) -> Path:
    """Read every record under ``root`` and write the summary table.

    Parameters
    ----------
    root
        A results directory (any depth of record directories below it).
    output
        The table's path; by default ``root / "summary.csv"``.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    records = [(meta.parent, read_record(meta.parent)) for meta in sorted(root.rglob("meta.json"))]
    rows = summary_rows(records, root)
    extra = sorted({key for row in rows for key in row} - set(IDENTITY))
    target = output or root / "summary.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*IDENTITY, *extra], restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def _parse(text: str) -> Any:
    """Parse one CSV cell back to ``bool``, ``int`` or ``float`` where it is one."""
    if text in ("True", "False"):
        return text == "True"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


#: The identity columns that are always text: a configuration hash such as ``596353258303e911``
#: would otherwise read as a number.
_TEXT: Final = frozenset(IDENTITY) - {"n", "time_total", "peak_memory_gib"}


def read_summary(path: Path) -> list[dict[str, Any]]:
    """Read a ``summary.csv`` written by :func:`collect`, with numbers and flags parsed.

    Lists stay as their ``;``-joined text, and empty cells as ``""``; the identity columns other
    than ``n``, ``time_total`` and ``peak_memory_gib`` are always text.

    Examples
    --------
    >>> _parse("3"), _parse("1e-12"), _parse("nan"), _parse("True"), _parse("R2a")
    (3, 1e-12, nan, True, 'R2a')
    """
    with path.open(newline="") as stream:
        return [
            {key: value if key in _TEXT else _parse(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def read_arrays(root: Path, row: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """Return the NPZ arrays of the record behind one summary row (``root`` as collected)."""
    with np.load(root / str(row["directory"]) / "arrays.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


#: The columns of a row that name its run's configuration beside its case (problem, variant and
#: resolution): every option a campaign varies, and not the initial-state candidates tried.
_CONFIGURATION = (
    "regime",
    "qubits",
    "carleman",
    "layout",
    "mult",
    "truncation",
    "rescale",
    "label",
    "tier",
    "initial_kind",
    "epsilon",
    "filter",
    "encoder",
    "verify_mode",
)

#: How a report names each preparation filter (:data:`pihm.qsp.prepare.FilterKind`).
_FILTER_NAMES: Final[Mapping[str, str]] = {
    "qite": "the QITE filter at the imaginary time the gap sets",
    "minimax": "the minimax filter at the degree that suppresses the excited spectrum to ε",
}


def filter_of(row: Mapping[str, Any]) -> str:
    """Return the filter a Pillar 2 row ran: its ``filter`` column, or QITE where it has none.

    A record names its filter only when it is not QITE, as its identity does
    (:class:`pihm.pipeline.spec.RunSpec`).
    """
    return str(row.get("filter") or "qite")


def encoder_of(row: Mapping[str, Any]) -> str:
    """Return the standard form's encoder of 𝔾 a row ran: its ``encoder``, or the structured one.

    A record names its encoder only when it is the published one, as its identity does
    (:class:`pihm.pipeline.spec.RunSpec`).
    """
    return str(row.get("encoder") or "structured")


def filter_prose(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return the filters the ``ok`` rows ran, named for a report's prose."""
    names = sorted({filter_of(row) for row in rows if row.get("status") == "ok"}) or ["qite"]
    return " and ".join(_FILTER_NAMES[name] for name in names)


def candidate_rank(row: Mapping[str, Any]) -> tuple[float, int]:
    """Return the order in which :func:`keep_best_candidate` prefers one ``ok`` row to another.

    The larger measured success probability wins, and an exact tie goes to the candidate a single
    run tries first (:data:`pihm.qsp.initial.GEOMETRIC_R`'s order). ``row`` is a summary row or a
    record's metrics.
    """
    r = row.get("geometric_r")
    first = r[0] if isinstance(r, list) else r
    tried = GEOMETRIC_R.index(first) if first in GEOMETRIC_R else len(GEOMETRIC_R)
    return float(row["success_probability"]), -tried


def keep_best_candidate(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return ``rows`` with one ``ok`` row per case: the candidate of largest success probability.

    A preparation tries the geometric initial states and keeps the one with the larger
    measured post-selection probability. A campaign may run each candidate as its own record;
    this makes that choice across the records, as a single run makes it across its candidates
    (:func:`candidate_rank`), among rows that differ in nothing else. Rows that are not ``ok``
    (failed, infeasible) are all kept, in their original order.
    """
    kept: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...] | int] = []
    others: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("status") != "ok":
            order.append(index)
            others[index] = dict(row)
            continue
        case = tuple(
            tuple(value) if isinstance(value, list) else value
            for value in (
                row.get(column) for column in ("problem", "variant", "n", *_CONFIGURATION)
            )
        )
        if case not in kept:
            order.append(case)
            kept[case] = dict(row)
        elif candidate_rank(row) > candidate_rank(kept[case]):
            kept[case] = dict(row)
    return [others[key] if isinstance(key, int) else kept[key] for key in order]


def cases(rows: Iterable[Mapping[str, Any]], **match: Any) -> dict[Case, Mapping[str, Any]]:
    """Return the ``ok`` rows whose columns equal every ``match``, one per case.

    Raises
    ------
    ValueError
        If two such rows of one case differ in their configuration or their initial-state
        candidates: the rows do not say which is meant, so ``match`` (or a selection such as
        :func:`keep_best_candidate`) must.
    """
    found: dict[Case, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or any(row.get(k) != v for k, v in match.items()):
            continue
        case = (str(row["problem"]), str(row["variant"]), int(row["n"]))
        first = found.setdefault(case, row)
        differ = [c for c in (*_CONFIGURATION, "geometric_r") if first.get(c) != row.get(c)]
        if differ:
            raise ValueError(f"two rows of case {case} differ in {differ}: select one by them")
    return found


def side_by_side(
    forms: Mapping[str, Mapping[Case, Mapping[str, Any]]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Return the cases every form has, ordered, with each column suffixed by its form's name.

    Examples
    --------
    >>> a = {("R1", "corrected", 3): {"degree": 365}}
    >>> b = {("R1", "corrected", 3): {"degree": 257}, ("R1", "corrected", 4): {"degree": 383}}
    >>> (row,) = side_by_side({"standard": a, "descriptor": b}, ["degree"])
    >>> row["n"], row["degree_standard"], row["degree_descriptor"]
    (3, 365, 257)
    """
    shared = set.intersection(*(set(rows) for rows in forms.values())) if forms else set()
    table = []
    for case in sorted(shared):
        entry: dict[str, Any] = {"problem": case[0], "variant": case[1], "n": case[2]}
        for form, rows in forms.items():
            entry |= {f"{column}_{form}": rows[case].get(column) for column in columns}
        table.append(entry)
    return table


def encoding_comparison(
    standard: tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    descriptor: tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    r"""Return both forms' encodings of every case each has encoded, side by side.

    Each form gives its ``"encode"`` rows and the ``"build"`` rows of the same residuals, from
    which :math:`\|R\| = \sqrt{\|H\|}`. A form's looseness is :math:`\alpha_R/\|R\|`, the factor
    by which the walk's queries exceed an ideal encoding's; it is NaN where the case has no build.

    Returns
    -------
    list of dict
        Per case, ordered by problem and n: ``problem``, ``variant``, ``n``, and for each form
        (suffix ``_standard``, ``_descriptor``) ``alpha_R``, ``looseness``, ``ancilla`` and
        ``cx``.
    """
    forms = {"standard": standard, "descriptor": descriptor}
    encodes = {form: cases(pair[0]) for form, pair in forms.items()}
    norms = {form: cases(pair[1]) for form, pair in forms.items()}
    shared = sorted(set(encodes["standard"]) & set(encodes["descriptor"]))
    table = []
    for case in shared:
        entry: dict[str, Any] = {"problem": case[0], "variant": case[1], "n": case[2]}
        for form in forms:
            encoded = encodes[form][case]
            build = norms[form].get(case)
            alpha = float(encoded["alpha_R"])
            norm = math.sqrt(float(build["hamiltonian_norm"])) if build else math.nan
            entry |= {
                f"alpha_R_{form}": alpha,
                f"looseness_{form}": alpha / norm,
                f"ancilla_{form}": int(encoded["n_anc_qubits"]),
                f"cx_{form}": int(encoded["cx_count"]),
            }
        table.append(entry)
    return table
