"""The collected results by campaign name, and the selections of them that the analysis reads.

``examples/campaigns.yaml`` names each campaign's results directory, so every analysis reads the
same records, and a rerun into new directories is one change there. :class:`Results` finds a
campaign's
``summary.csv`` by name.

The selections name the records an argument is built on, so that no analysis re-derives
them:

* :func:`primary_preparations`: each panel's primary state preparation (tier T2);
* :func:`reference_builds`: each case's classical build, against which an encoding's looseness is
  measured.

:func:`cost_comparison` sets both forms' resource estimates of each case side by side, as ratios;
:func:`encoding_scaling` gives each encoding's degree with the two ratios that set it, and
:func:`degree_exponents` the degree's power law in :math:`N`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from pihm.pipeline.records import source_digest
from pihm.pipeline.summary import Case, cases, read_summary
from pihm.resources import fit_power_law

__all__ = [
    "AGREEMENT_BAR",
    "CONFIG",
    "NOT_PRIMARY",
    "PHASES_BAR",
    "PROBABILITY_BAR",
    "Results",
    "cost_comparison",
    "degree_exponents",
    "encoding_scaling",
    "primary_preparations",
    "reference_builds",
]

#: The file that names each campaign's results directory, relative to the pihm directory.
CONFIG: Final = Path("examples") / "campaigns.yaml"

#: The "T1 vs T2" bar, on the circuit's output before normalising (from a unit
#: input) and on the prepared state.
AGREEMENT_BAR: Final = 1e-10

#: The bar on the measured success probability against the predicted, relative.
PROBABILITY_BAR: Final = 1e-8

#: The bar on the realised polynomial against its target.
PHASES_BAR: Final = 1e-12

#: The labels of state-preparation records that are not a panel's primary run. They mark one
#: initial-state candidate of a run (for T1's predictions), an initial-state control, and the
#: doubled space's projection.
NOT_PRIMARY: Final = frozenset({"per candidate", "initial-state control", "doubled space"})


@dataclass(frozen=True)
class Results:
    """Each campaign's results directory, by campaign name.

    Attributes
    ----------
    directories
        The results directory of each campaign.
    labels
        The run-key segment of each run label (:func:`pihm.pipeline.catalogue.run_key`), as the
        keys write them.
    sources
        The earlier sources whose records stand beside the tree's own, by source digest
        (:func:`pihm.pipeline.records.source_digest`), each with why: a source that differs from
        the tree only where none of its records could.
    """

    directories: Mapping[str, Path]
    labels: Mapping[str, str] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def read(cls, config: Path, root: Path) -> Results:
        """Read the campaigns of ``config``, whose directories are relative to ``root``."""
        data = yaml.safe_load(config.read_text())
        return cls(
            {str(name): (root / str(path)).resolve() for name, path in data["campaigns"].items()},
            {str(label): str(token) for label, token in (data.get("labels") or {}).items()},
            {str(digest): str(why) for digest, why in (data.get("sources") or {}).items()},
        )

    @classmethod
    def at(cls, root: Path) -> Results:
        """Read :data:`CONFIG` under the pihm directory ``root``."""
        return cls.read(root / CONFIG, root)

    def accepted_sources(self) -> frozenset[str]:
        """Return the source digests whose records stand: the tree's own and :attr:`sources`."""
        return frozenset({source_digest(), *self.sources})

    def directory(self, campaign: str) -> Path:
        """Return ``campaign``'s results directory.

        Raises
        ------
        KeyError
            If :data:`CONFIG` does not name ``campaign``.
        """
        if campaign not in self.directories:
            raise KeyError(f"{campaign} is not a campaign of {CONFIG}")
        return self.directories[campaign]

    def missing(self, campaigns: Iterable[str]) -> tuple[str, ...]:
        """Return the campaigns, of ``campaigns``, that have no collected ``summary.csv``."""
        return tuple(c for c in campaigns if not (self.directory(c) / "summary.csv").exists())

    def rows(self, campaign: str, **match: Any) -> list[dict[str, Any]]:
        """Return ``campaign``'s summary rows whose columns equal ``match``.

        The rows are :func:`pihm.pipeline.summary.read_summary`'s. A campaign of both forms is
        read one form at a time with ``regime="standard"`` or ``regime="descriptor"``.
        """
        rows = read_summary(self.directory(campaign) / "summary.csv")
        return [row for row in rows if all(row.get(k) == v for k, v in match.items())]

    def describe(self, campaigns: Iterable[str]) -> str:
        """Return one line per campaign: its record count and how many are ``ok``, or missing."""
        lines = []
        for campaign in campaigns:
            if self.missing([campaign]):
                lines.append(f"{campaign}: no results in {self.directory(campaign)}")
                continue
            rows = self.rows(campaign)
            ok = sum(row["status"] == "ok" for row in rows)
            lines.append(f"{campaign}: {len(rows)} records, {ok} ok")
        return "\n".join(lines)


def primary_preparations(
    rows: Iterable[Mapping[str, Any]], **match: Any
) -> list[Mapping[str, Any]]:
    """Return each panel's primary state preparations whose columns equal ``match``.

    A primary preparation, in either form, is an ``ok`` T2 record from the geometric initial
    state of the untruncated residual, and not one of :data:`NOT_PRIMARY`'s
    labels. The standard form's primary runs from R2 on are labelled as the paper's own
    construction, so the labels are excluded, not required to be empty.

    Examples
    --------
    >>> row = {"status": "ok", "tier": "T2", "initial_kind": "geometric", "truncation": "none"}
    >>> rows = [row | {"label": ""}, row | {"label": "per candidate"}]
    >>> len(primary_preparations(rows))
    1
    """
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("tier") == "T2"
        and row.get("initial_kind") == "geometric"
        and row.get("truncation") == "none"
        and row.get("label") not in NOT_PRIMARY
        and all(row.get(column) == value for column, value in match.items())
    ]


def reference_builds(rows: Iterable[Mapping[str, Any]]) -> dict[Case, Mapping[str, Any]]:
    """Return each case's reference build: the unlabelled, untruncated and lifted assembly.

    In the standard form that is the paper's assembly; in the descriptor, the block-rescaled one
    (its unscaled ablation is labelled).
    """
    return cases(rows, label="", truncation="none", mult="lift")


def _ratio(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> float:
    """Return ``a[key] / b[key]``, or NaN where either record predates the metric."""
    if a.get(key) in (None, "") or b.get(key) in (None, ""):
        return float("nan")
    return float(a[key]) / float(b[key])


def _step(row: Mapping[str, Any]) -> float:
    """Return a T3 row's cx per walk step: its total over the preparations and the degree."""
    return float(row["gate_count"]) / (float(row["preparations"]) * float(row["degree"]))


def cost_comparison(
    standard: tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    descriptor: tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    r"""Return, per case both forms estimated, the standard form's cost over the descriptor's.

    Each form gives its T3 rows and its build rows. A preparation's total cx is the product of
    the preparations amplitude amplification applies, the filter's degree and the cx of one walk
    step (:func:`pihm.qsp.prepare.t3`); each ratio is the standard form's over the descriptor's,
    so a ratio above one means the descriptor costs less. ``ideal_degree`` compares the degrees
    at :math:`\alpha_R = \|R\|` in both forms, with neither encoding's looseness in them.

    Returns
    -------
    list of dict
        Per case, ordered by problem and n: ``problem``, ``variant``, ``n``, the ratios ``total``,
        ``preparations``, ``degree``, ``ideal_degree`` (NaN where a record predates it) and
        ``step``, and ``trusted``, whether both forms' reference builds are trusted (false where
        either form has no build of the case).
    """
    estimates = {"standard": cases(standard[0]), "descriptor": cases(descriptor[0])}
    builds = {
        "standard": reference_builds(standard[1]),
        "descriptor": reference_builds(descriptor[1]),
    }
    table = []
    for case in sorted(set(estimates["standard"]) & set(estimates["descriptor"])):
        a, b = estimates["standard"][case], estimates["descriptor"][case]
        trusted = all(builds[form].get(case, {}).get("trusted") is True for form in builds)
        table.append(
            {
                "problem": case[0],
                "variant": case[1],
                "n": case[2],
                "total": float(a["gate_count"]) / float(b["gate_count"]),
                "preparations": float(a["preparations"]) / float(b["preparations"]),
                "degree": float(a["degree"]) / float(b["degree"]),
                "ideal_degree": _ratio(a, b, "ideal_degree"),
                "step": _step(a) / _step(b),
                "trusted": trusted,
            }
        )
    return table


def encoding_scaling(
    encodings: Mapping[
        str,
        tuple[
            Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]
        ],
    ],
    **match: Any,
) -> list[dict[str, Any]]:
    r"""Return, per encoding and case whose columns equal ``match``, its degree and what sets it.

    Each encoding gives its T3 rows, the builds of the form whose Hamiltonian it encodes, and that
    form's primary T2 rows (:func:`primary_preparations`; none where T2 cannot run the encoding).
    The filter's degree grows as :math:`\sqrt{\alpha_H/\Delta}` with :math:`\alpha_H=\alpha_R^2`,
    the product of two ratios: the encoding's looseness
    :math:`\alpha_H/\|H\|`, and the Hamiltonian's relative gap :math:`\Delta/\|H\|`.

    Returns
    -------
    list of dict
        Per encoding, then case: ``encoding``, ``problem``, ``variant``, ``n``, ``looseness``,
        ``relative_gap``, ``degree`` and ``cx`` (T3's degree and total cx), ``measured_degree``
        (T2's, or ``None``) and ``trusted`` (the build's reference). A case without a build is
        left out.
    """
    table = []
    for name, (estimates, builds, prepared) in encodings.items():
        norms = reference_builds(builds)
        measured = cases(prepared)
        for case, row in sorted(cases(estimates, **match).items()):
            build = norms.get(case)
            if build is None:
                continue
            norm = float(build["hamiltonian_norm"])
            twin = measured.get(case)
            table.append(
                {
                    "encoding": name,
                    "problem": case[0],
                    "variant": case[1],
                    "n": case[2],
                    "looseness": float(row["alpha_R"]) ** 2 / norm,
                    "relative_gap": float(row["gap"]) / norm,
                    "degree": float(row["degree"]),
                    "cx": float(row["gate_count"]),
                    "measured_degree": float(twin["degree"]) if twin else None,
                    "trusted": build.get("trusted") is True,
                }
            )
    return table


def degree_exponents(
    table: Iterable[Mapping[str, Any]], ns: Sequence[int], column: str = "degree"
) -> list[dict[str, Any]]:
    """Return, per encoding and problem of :func:`encoding_scaling`'s table, a cost's exponent.

    The exponent is the least-squares power law of ``column`` (T3's degree, or its total ``cx``)
    in :math:`N=2^n` over the resolutions ``ns`` the table has
    (:func:`pihm.resources.fit_power_law`), at least two.

    Returns
    -------
    list of dict
        Per encoding, then problem: ``encoding``, ``problem``, ``exponent``, ``ns`` (the
        resolutions fitted) and ``trusted`` (every reference in the fit).
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in table:
        if int(row["n"]) in ns:
            groups.setdefault((str(row["encoding"]), str(row["problem"])), []).append(row)
    fits = []
    for (encoding, problem), points in groups.items():
        if len(points) < 2:
            continue
        points = sorted(points, key=lambda row: int(row["n"]))
        _, exponent = fit_power_law(
            [2.0 ** int(row["n"]) for row in points], [float(row[column]) for row in points]
        )
        fits.append(
            {
                "encoding": encoding,
                "problem": problem,
                "exponent": exponent,
                "ns": [int(row["n"]) for row in points],
                "trusted": all(row["trusted"] for row in points),
            }
        )
    return fits
