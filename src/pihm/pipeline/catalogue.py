r"""The results catalogue: every run the campaigns expect, under a readable key, with its outcome.

A campaign names the runs it expects (:func:`~pihm.pipeline.campaign.load_campaign`), and its
results directory holds the records that exist. The catalogue joins the two, so that a run that
has not happened yet is reported as such rather than silently missing. Analyses quote
numbers from it.

Keys
----
A run's key reads like its record directory without the hash (:func:`run_key`)::

    <stage>/<problem>[/faithful]/<regime>/n<size>[/<tier>][/<option>=<value>...][/<label>]

for example ``stateprep/R2a/standard/n3/T2``, or ``build/R2a/faithful/standard/n3`` for the panel as
the paper printed it, which only the reproduction runs. The tier is written for the
``"stateprep"`` and ``"readout"`` stages; an option only when it differs from
:class:`~pihm.pipeline.spec.RunSpec`'s default (``carleman=3``, ``layout=chain``,
``truncation=tau``, ``verify=none``, ``filter=minimax``, ``encoder=published``); a label as its
token (``exact-root``, or nothing). A key never names the initial-state candidates: the runs of one
case that differ only in ``geometric_r`` share a key, and the entry keeps the best of them
(:func:`~pihm.pipeline.summary.candidate_rank`).

Statuses
--------
``ok``, ``infeasible``, ``failed``
    The outcome of the configuration the campaign now expects. An infeasible run keeps its
    resource estimate among its values.
``stale``
    No record of the expected configuration, but one of an earlier configuration under the same
    key (a problem, an option or the run schema has changed since), or a record no campaign
    expects any more. Its values are kept, and marked stale by whoever quotes them.
``pending``
    The campaign expects the run, and there is no record under its key.

Examples
--------
>>> label_token("x_s = -0.5"), label_token("tau ablation")
('x-s--0.5', 'tau-ablation')
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from pihm.pipeline.campaign import Campaign
from pihm.pipeline.records import JSONValue, Record, read_record
from pihm.pipeline.spec import RunSpec
from pihm.pipeline.summary import candidate_rank

__all__ = [
    "Entry",
    "EntryStatus",
    "Source",
    "catalogue",
    "label_token",
    "run_key",
]

#: The outcome of a key (see the module docstring).
EntryStatus: TypeAlias = Literal["ok", "infeasible", "failed", "stale", "pending"]

#: Run options written into a key when they differ from :class:`RunSpec`'s default, in this order.
_KEY_OPTIONS: tuple[str, ...] = (
    "carleman",
    "layout",
    "lift",
    "mult",
    "truncation",
    "verify",
    "resources",
    "kind",
    "epsilon",
    "filter",
    "encoder",
    "shots",
)

#: :class:`RunSpec`'s defaults, which also stand in for the fields an older record lacks.
_DEFAULTS: dict[str, JSONValue] = {
    item.name: item.default
    for item in dataclasses.fields(RunSpec)
    if item.default is not dataclasses.MISSING
}

#: The stages whose key names the tier.
_TIERED = ("stateprep", "readout")

#: The characters a key segment may hold.
_SEGMENT = re.compile(r"[A-Za-z0-9.=+\-]+")


@dataclass(frozen=True)
class Source:
    """The record behind (part of) an entry.

    Attributes
    ----------
    config_hash
        The record's configuration hash.
    directory
        The record directory, relative to its campaign's results directory.
    git_commit
        The commit the record was produced at, if known.
    git_dirty
        Whether that working tree had uncommitted changes, if known.
    source_digest
        The digest of the source that made the record
        (:func:`pihm.pipeline.records.source_digest`), if known.
    """

    config_hash: str
    directory: str
    git_commit: str | None
    git_dirty: bool | None
    source_digest: str | None = None


@dataclass(frozen=True)
class Entry:
    """What the catalogue knows about one key.

    Attributes
    ----------
    key
        The run key (:func:`run_key`), or a derived number's own key.
    status
        The key's outcome.
    reason
        Why the status is what it is, in words; empty for ``ok``.
    values
        The numeric scalar results, by metric name, finite only; the run's wall time is
        ``time_total`` and its peak memory ``peak_memory_gib``.
    flags
        The Boolean results, by metric name.
    sources
        The records the values come from.
    """

    key: str
    status: EntryStatus
    reason: str = ""
    values: Mapping[str, int | float] = field(default_factory=dict)
    flags: Mapping[str, bool] = field(default_factory=dict)
    sources: tuple[Source, ...] = ()


def label_token(label: str) -> str:
    """Return the key segment a label becomes when no token is given for it.

    Lower case, with every run of other characters than letters, digits and ``.``, ``+`` and
    ``-`` turned into one ``-``.
    """
    return re.sub(r"[^a-z0-9.+\-]+", "-", label.lower()).strip("-")


def _option(value: JSONValue) -> str:
    """Write an option's value as a key segment."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def run_key(spec: Mapping[str, JSONValue], labels: Mapping[str, str] | None = None) -> str:
    """Return the key of a run, from its specification.

    The specification is :meth:`~pihm.pipeline.spec.RunSpec.to_dict`'s, or a record's.

    Parameters
    ----------
    spec
        The run specification.
    labels
        A key segment for each run label; an empty token leaves the label out of the key, and a
        label without a token becomes :func:`label_token`'s.

    Raises
    ------
    ValueError
        If a label's token holds a character a key segment may not.
    """
    problem = spec["problem"]
    assert isinstance(problem, Mapping)
    size = "x".join(str(k) for k in spec["n"])
    parts = [str(spec["stage"]), str(problem["name"])]
    if problem.get("variant") == "faithful":
        parts.append("faithful")
    parts += [str(spec["regime"]), f"n{size}"]
    if spec["stage"] in _TIERED:
        parts.append(str(spec.get("tier", _DEFAULTS["tier"])))
    for option in _KEY_OPTIONS:
        value = spec.get(option, _DEFAULTS[option])
        if value != _DEFAULTS[option]:
            parts.append(f"{option}={_option(value)}")
    label = str(spec.get("label", ""))
    token = (labels or {}).get(label, label_token(label))
    if token and not _SEGMENT.fullmatch(token):
        raise ValueError(f"the token {token!r} of label {label!r} is not a key segment")
    if token:
        parts.append(token)
    return "/".join(parts)


def _identity(spec: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Return ``spec`` without the initial-state candidates, which a key does not name."""
    return {key: value for key, value in spec.items() if key != "geometric_r"}


def _scalars(record: Record) -> tuple[dict[str, int | float], dict[str, bool]]:
    """Return a record's finite numeric results and its Boolean ones."""
    values: dict[str, int | float] = {}
    flags: dict[str, bool] = {}
    for name, value in record.metrics.items():
        if isinstance(value, bool):
            flags[name] = value
        elif isinstance(value, (int, float)) and math.isfinite(value):
            values[name] = value
    total = record.timings.get("total")
    if total is not None and math.isfinite(total):
        values.setdefault("time_total", total)
    if record.peak_memory_gib is not None:
        values.setdefault("peak_memory_gib", record.peak_memory_gib)
    return values, flags


@dataclass(frozen=True)
class _Found:
    """A record read from a results directory."""

    record: Record
    source: Source


def _best(found: Sequence[_Found]) -> _Found:
    """Return the candidate to quote: the best ``ok`` one, else the first."""
    ok = [item for item in found if item.record.status == "ok"]
    if not ok:
        return found[0]
    if len(ok) == 1 or any("success_probability" not in item.record.metrics for item in ok):
        return ok[0]
    return max(ok, key=lambda item: candidate_rank(item.record.metrics))


def _entry(key: str, status: EntryStatus, reason: str, chosen: Sequence[_Found]) -> Entry:
    """Return the entry quoting the best of ``chosen`` (no values when it has no usable one)."""
    if not chosen or status in ("failed", "pending"):
        return Entry(key, status, reason, sources=tuple(item.source for item in chosen))
    best = _best(chosen)
    if status == "stale" and best.record.status != "ok":
        return Entry(key, status, reason, sources=(best.source,))
    values, flags = _scalars(best.record)
    return Entry(key, status, reason, values, flags, (best.source,))


def _outcome(key: str, found: Sequence[_Found]) -> Entry:
    """Return the entry of a key whose every expected candidate has a record."""
    statuses = [item.record.status for item in found]
    if "ok" in statuses:
        return _entry(key, "ok", "", found)
    infeasible = [item for item in found if item.record.status == "infeasible"]
    if infeasible:
        reasons = infeasible[0].record.metrics.get("infeasible_reasons") or []
        text = "; ".join(str(r) for r in reasons) if isinstance(reasons, list) else str(reasons)
        return _entry(key, "infeasible", text or "beyond its resource budget", infeasible)
    error = (found[0].record.error or "").strip().splitlines()
    return _entry(key, "failed", error[-1] if error else "the run failed", found)


def _read(root: Path) -> list[_Found]:
    """Read every record under ``root`` (none when it does not exist)."""
    found = []
    for meta in sorted(root.rglob("meta.json")):
        record = read_record(meta.parent)
        provenance = record.provenance
        source = Source(
            record.config_hash,
            str(meta.parent.relative_to(root)),
            provenance.git_commit,
            provenance.git_dirty,
            provenance.source_digest,
        )
        found.append(_Found(record, source))
    return found


def catalogue(
    campaigns: Iterable[tuple[Campaign, Path]], labels: Mapping[str, str] | None = None
) -> dict[str, Entry]:
    """Return the entry of every run the campaigns expect, and of every stale record, by key.

    Parameters
    ----------
    campaigns
        Each campaign with its results directory, which need not exist yet.
    labels
        The key segment of each run label (see :func:`run_key`).

    Raises
    ------
    ValueError
        If two runs that differ in more than their initial-state candidates share a key: give one
        of them a label, or a label token.
    """
    expected: dict[str, dict[str, RunSpec]] = {}
    owner: dict[str, str] = {}
    records: dict[str, _Found] = {}
    for campaign, root in campaigns:
        for spec in campaign.specs:
            key = run_key(spec.to_dict(), labels)
            runs = expected.setdefault(key, {})
            for other in runs.values():
                if _identity(other.to_dict()) != _identity(spec.to_dict()):
                    raise ValueError(
                        f"two runs of campaign {campaign.name} share the key {key}; "
                        "distinguish them with a label"
                    )
            runs[spec.config_hash] = spec
            owner.setdefault(key, campaign.name)
        for item in _read(root):
            records.setdefault(item.source.config_hash, item)

    wanted = {digest for runs in expected.values() for digest in runs}
    earlier: dict[str, list[_Found]] = {}
    for digest, item in records.items():
        if digest not in wanted:
            earlier.setdefault(run_key(item.record.spec, labels), []).append(item)

    entries: dict[str, Entry] = {}
    for key, runs in expected.items():
        found = [records[digest] for digest in runs if digest in records]
        found = [item for item in found if item.record.status != "skipped"]
        if len(found) == len(runs):
            entries[key] = _outcome(key, found)
        elif key in earlier:
            old = ", ".join(item.source.config_hash for item in earlier[key])
            reason = f"recorded for an earlier configuration ({old}); rerun {owner[key]}"
            entries[key] = _entry(key, "stale", reason, earlier[key])
        else:
            done = f"{len(found)} of {len(runs)} candidate runs recorded; " if found else ""
            entries[key] = Entry(key, "pending", f"{done}not yet run ({owner[key]})")
    for key, items in earlier.items():
        if key not in entries:
            entries[key] = _entry(key, "stale", "no campaign expects this run any more", items)
    return entries
