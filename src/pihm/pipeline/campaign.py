r"""Campaigns: YAML lists of runs, expanded into run specifications and executed.

A campaign file holds a name, a description and a list of **groups**. Each group is a grid:
every listed problem × variant × resolution, with shared options::

    name: p2_reproduction
    description: The paper's panels in the standard form.
    groups:
      - problems: [R2]              # a rung, or problem names such as R2a
        variants: [faithful, corrected]    # the paper's panels as printed, then as corrected,
                                           # for the reproduction only; default the problems
        n: {from: 2, to: 10}        # or a list, such as [3, 4]
        carleman: {from: 2, to: 6}  # a Carleman lift's orders, as n; each order with every n
        truncation: tau             # assembly options: regime, mult, truncation, rescale,
                                    # layout (chain: the descriptor's first-order ablation)
        stage: encode                       # "build", "encode", "stateprep" or
                                            # "readout"
        verify: exact                       # encode only: exact, probe or none
        resources: false                    # encode only: whether to transpile and count gates
        kind: geometric                     # stateprep only: geometric, uniform or random
        geometric_r: [0.5]                  # geometric only: the r tried (default 0.5 and -0.5)
        epsilon: 1.0e-6                     # stateprep only: the QITE filter's target accuracy
        encoder: published                  # standard-form T3 only: the paper's G (default
                                            # structured)
        tier: T1                            # stateprep, readout: T2 (default) or T1, the circuit
        shots: 100000                       # readout T1 only: also sample the probabilities
        label: tau ablation         # groups runs in reports; part of the hash
        memory: 32                  # a Slurm array task's allocation, GB and hours (default
        hours: 24                   # 64 and 12); never part of the hash
        modify:                     # optional changes to the ladder problem (the triage)
          constraints: corrected    # take the constraints of the other variant
          source: maclaurin:11      # or chebyshev: re-represent every source
          regular: [0.5]            # move the regular point

A build group with several Carleman orders must sweep ``n`` too: the lift's error
is only read against the time axis's resolution. Other stages may hold ``n`` fixed, as a
circuit's resources against the order do.

A group's ``memory`` and ``hours`` are its runs' :class:`Allocation` as Slurm array tasks, one
array per distinct allocation: what a run
needs from the cluster, not what it computes, so they stay out of its identity. They come from the
runs' own records, each of which keeps its wall time and peak memory.

The executor is idempotent: a run whose record exists with status ``ok`` is skipped. Editing a
problem changes its digest and so its records' hashes; :func:`prune` then removes the records the
campaign no longer contains, so that the summary holds one row per run.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args

import yaml

from pihm.assemble import Regime
from pihm.pipeline.records import read_record
from pihm.pipeline.spec import RunSpec, record_directory
from pihm.pipeline.stages import run
from pihm.problems import ladder
from pihm.problems.spec import Equation, Problem, Representation, Source, Term

__all__ = [
    "Allocation",
    "Campaign",
    "classes",
    "expand",
    "load_campaign",
    "modify",
    "prune",
    "run_campaign",
]

logger = logging.getLogger(__name__)

_OPTIONS = (
    "stage",
    "regime",
    "mult",
    "truncation",
    "rescale",
    "layout",
    "lift",
    "verify",
    "resources",
    "kind",
    "geometric_r",
    "epsilon",
    "filter",
    "encoder",
    "tier",
    "shots",
    "label",
)
_ALLOCATION_KEYS = ("memory", "hours")
_GROUP_KEYS = frozenset(
    {"problems", "variants", "n", "carleman", "modify", *_OPTIONS, *_ALLOCATION_KEYS}
)


@dataclass(frozen=True, order=True)
class Allocation:
    """What one run asks of the cluster as a Slurm array task.

    Attributes
    ----------
    memory
        Gigabytes.
    hours
        Wall time.
    """

    memory: int = 64
    hours: int = 12


@dataclass(frozen=True)
class Campaign:
    """A named list of runs.

    Attributes
    ----------
    name
        The campaign's name, used for its results directory.
    description
        What the campaign is for.
    specs
        The expanded runs, in file order.
    allocations
        Each run's allocation, in the same order.
    """

    name: str
    description: str
    specs: tuple[RunSpec, ...]
    allocations: tuple[Allocation, ...] = ()


def modify(problem: Problem, changes: Mapping[str, Any]) -> Problem:
    """Return ``problem`` with the triage's ``changes`` applied (see the module documentation).

    Raises
    ------
    ValueError
        For an unknown change.
    """
    unknown = set(changes) - {"constraints", "source", "regular"}
    if unknown:
        raise ValueError(f"unknown modifications {sorted(unknown)}")
    if "constraints" in changes:
        donor = ladder.problem(problem.name, changes["constraints"])
        problem = dataclasses.replace(problem, constraints=donor.constraints)
    if "source" in changes:
        kind, _, degree = str(changes["source"]).partition(":")
        if kind not in get_args(Representation):
            raise ValueError(f"unknown source representation {kind!r}")
        representation = cast(Representation, kind)

        def represent(term: Term) -> Term:
            if term.source is None:
                return term
            order = int(degree) if degree else None
            source = Source(term.source.expression, representation, order)
            return dataclasses.replace(term, source=source)

        problem = dataclasses.replace(
            problem,
            equations=tuple(
                Equation(tuple(represent(t) for t in eq.terms), eq.label)
                for eq in problem.equations
            ),
        )
    if "regular" in changes:
        at = tuple(float(value) for value in changes["regular"])
        problem = dataclasses.replace(problem, regular=dataclasses.replace(problem.regular, at=at))
    return problem


def _resolutions(value: Any) -> list[int]:
    """Return the list of resolutions of a group: a list, or ``{from: a, to: b}`` inclusive."""
    if isinstance(value, Mapping):
        return list(range(int(value["from"]), int(value["to"]) + 1))
    if isinstance(value, (int, list)):
        return [int(value)] if isinstance(value, int) else [int(k) for k in value]
    raise ValueError(f"n must be a list or {{from, to}}, got {value!r}")


def expand(groups: Sequence[Mapping[str, Any]]) -> list[RunSpec]:
    """Expand campaign groups into run specifications (problems × variants × orders × resolutions).

    Raises
    ------
    ValueError
        For an unknown key, problem, rung or variant, and for a build's sweep over Carleman orders
        at a single resolution.
    """
    specs: list[RunSpec] = []
    for group in groups:
        unknown = set(group) - _GROUP_KEYS
        if unknown:
            raise ValueError(f"unknown campaign keys {sorted(unknown)}")
        names: list[str] = []
        for entry in group["problems"]:
            names.extend(ladder.names(entry) if entry in ladder.RUNGS else [entry])
        options = {key: group[key] for key in _OPTIONS if key in group}
        if "geometric_r" in options:
            options["geometric_r"] = tuple(float(r) for r in options["geometric_r"])
        if group.get("regime", "standard") not in get_args(Regime):
            raise ValueError(f"unknown regime {group['regime']!r}")
        resolutions = _resolutions(group["n"])
        orders: list[int | None] = (
            list(_resolutions(group["carleman"])) if "carleman" in group else [None]
        )
        stage = group.get("stage", "build")
        if stage == "build" and len(orders) > 1 and len(resolutions) < 2:
            raise ValueError("a sweep over Carleman orders needs one over n too")
        for name in names:
            for variant in group.get("variants", ["corrected"]):
                if variant == "faithful" and name not in ladder.PANELS:
                    continue  # R1 is not the paper's: a grid over R1 to R6 skips its faithful
                problem = modify(ladder.problem(name, variant), group.get("modify", {}))
                dimensions = len(problem.fields[0].axes)
                for order in orders:
                    for n in resolutions:
                        specs.append(RunSpec(problem, (n,) * dimensions, carleman=order, **options))
    return specs


def load_campaign(path: Path) -> Campaign:
    """Read a campaign file and expand it, with each run's allocation."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"{path} is not a campaign: it needs a 'groups' list")
    specs: list[RunSpec] = []
    allocations: list[Allocation] = []
    for group in data["groups"]:
        runs = expand([group])
        wanted = {key: int(group[key]) for key in _ALLOCATION_KEYS if key in group}
        specs += runs
        allocations += [Allocation(**wanted)] * len(runs)
    return Campaign(
        name=str(data.get("name", path.stem)),
        description=str(data.get("description", "")),
        specs=tuple(specs),
        allocations=tuple(allocations),
    )


def classes(campaign: Campaign) -> dict[Allocation, list[int]]:
    """Return the task numbers of each allocation the campaign's runs ask for, in task order."""
    found: dict[Allocation, list[int]] = {}
    for task, allocation in enumerate(campaign.allocations):
        found.setdefault(allocation, []).append(task)
    return found


def _done(directory: Path) -> bool:
    """Whether a record with status ``ok`` exists in ``directory``."""
    if not (directory / "meta.json").exists():
        return False
    try:
        return read_record(directory).status == "ok"
    except (OSError, ValueError, KeyError):
        return False


def run_campaign(specs: Iterable[RunSpec], root: Path, *, rerun: bool = False) -> list[Path]:
    """Run every specification whose ``ok`` record is not already under ``root``.

    Parameters
    ----------
    specs
        The runs.
    root
        The campaign's results directory.
    rerun
        Run everything, replacing existing records.

    Returns
    -------
    list of pathlib.Path
        The record directory of every specification, run or skipped.
    """
    directories = []
    for spec in specs:
        directory = record_directory(root, spec)
        if rerun or not _done(directory):
            record = run(spec, root)
            timing = record.timings.get("total", float("nan"))
            logger.info(
                "%s %s n=%s: %s in %.2f s",
                record.status,
                spec.problem.key,
                spec.n,
                spec.label or "-",
                timing,
            )
        directories.append(directory)
    return directories


def prune(specs: Iterable[RunSpec], root: Path) -> list[Path]:
    """Remove every record directory under ``root`` that is not one of ``specs``.

    Only directories holding a ``meta.json`` are touched, and only below ``root``: a record goes
    stale when its problem changes (a new digest, so a new hash) or leaves the campaign.

    Returns
    -------
    list of pathlib.Path
        The directories removed.
    """
    wanted = {record_directory(root, spec) for spec in specs}
    stale = sorted(meta.parent for meta in root.rglob("meta.json") if meta.parent not in wanted)
    for directory in stale:
        logger.info("pruning %s", directory.relative_to(root))
        shutil.rmtree(directory)
    return stale
