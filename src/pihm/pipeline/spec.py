r"""Run specifications: what a run depends on, and where its record lives.

A :class:`RunSpec` names everything a run depends on: the *whole* problem (not only its name,
so that the triage's derived problems are first-class and hash distinctly), the resolution, the
regime and the options of its stage. Its :attr:`~RunSpec.config_hash` is the record's identity,
and :func:`record_directory` where that record lives.

A paper panel's faithful variant exists for the standard form's comparison with the paper: the
descriptor form, which the paper does not have, never runs it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, TypeAlias

from pihm.assemble import FirstOrderLayout, Regime, Rescale, Truncation
from pihm.circuits.standard import StandardEncoder
from pihm.operators import MultiplicationMode
from pihm.pipeline.records import JSONValue, config_hash
from pihm.problems.spec import Problem
from pihm.qsp.filters import DEFAULT_EPSILON
from pihm.qsp.initial import GEOMETRIC_R, InitialKind
from pihm.qsp.prepare import FilterKind

__all__ = [
    "RUN_SCHEMA",
    "LiftLayout",
    "RunSpec",
    "Stage",
    "Tier",
    "Verify",
    "record_directory",
]

#: The stages of a run.
Stage: TypeAlias = Literal["build", "encode", "stateprep", "readout", "protocol", "filters", "lift"]

#: How deeply an ``"encode"`` or ``"lift"`` run checks its circuit: ``"ir"`` rebuilds a lift's
#: operator from its terms without the circuit (:mod:`pihm.pipeline.lift`).
Verify: TypeAlias = Literal["exact", "probe", "ir", "none"]

#: A ``"lift"`` run's layout of the Carleman lift (:mod:`pihm.pipeline.lift`).
LiftLayout: TypeAlias = Literal["symmetric", "tensor"]

#: Pillar 2's tiers, the circuit simulated (T1), the exact computation (T2), or the resource
#: estimate (T3, a ``"stateprep"`` run only).
Tier: TypeAlias = Literal["T1", "T2", "T3"]

#: Version of the run-specification layout inside a record's ``spec``.
RUN_SCHEMA = 1


@dataclass(frozen=True)
class RunSpec:
    r"""Everything one run depends on.

    Attributes
    ----------
    problem
        The problem, in full.
    n
        Qubits per axis of the field.
    stage
        The pipeline stage.
    regime
        The derivative representation.
    mult
        The multiplication mode passed to :func:`pihm.assemble.assemble`.
    truncation
        The row truncation passed to :func:`pihm.assemble.assemble`.
    rescale
        The descriptor's block rescaling passed to :func:`pihm.assemble.assemble`: ``None`` for
        the regime's default (``"block"`` in the descriptor), or ``"none"`` for the
        unscaled ablation. It enters a descriptor run's identity as the value it resolves to, and
        a standard-form run's not at all, whose state is one block.
    carleman
        The order :math:`K` of a Carleman lift passed to :func:`pihm.assemble.assemble`, or
        ``None`` for none. It enters the identity only when given, so every
        other record keeps its hash.
    layout
        How the descriptor carries a first-order equation or system, passed to
        :func:`pihm.assemble.assemble`: its mass form, or for a ``"build"`` run the
        ``"chain"`` ablation. It enters the identity only as the ablation.
    lift
        For a ``"lift"`` run only: the Carleman lift's layout, ``"symmetric"`` or ``"tensor"``
        (:mod:`pihm.pipeline.lift`). It enters the identity of a lift run only.
    verify
        For an ``"encode"`` or ``"lift"`` run: how deeply to check the circuit against the
        classical operator. ``"exact"`` needs a small total qubit count;
        ``"probe"`` applies it to random vectors (``"encode"``), ``"ir"`` rebuilds a lift from its
        terms without the circuit (``"lift"``), and ``"none"`` gives a construction-only,
        resource-only record.
    resources
        For an ``"encode"`` or ``"lift"`` run: whether to transpile the circuit and record its
        gate counts (:func:`pihm.resources.count_resources`), which can dominate the run time at
        larger n.
    kind
        For a ``"stateprep"`` run only: the initial state
        (:func:`pihm.qsp.prepare.t2`'s own ``kind``).
    geometric_r
        For a ``"stateprep"`` or ``"readout"`` run with ``kind="geometric"``: the :math:`r`
        values tried, keeping the larger measured success probability. A
        T1 campaign runs each candidate as its own record, and the reports keep, per case, the
        record with the larger probability (:func:`pihm.pipeline.summary.keep_best_candidate`).
    epsilon
        For a ``"stateprep"`` run only: the suppression the filter holds the excited spectrum to
        (:func:`pihm.qsp.prepare.t2`'s own ``epsilon``).
    filter
        For a ``"stateprep"`` run only: the preparation filter, ``"qite"`` or ``"minimax"``
        (:mod:`pihm.qsp.prepare`'s "The filter"). It enters the identity only when it is the
        minimax filter, so every record made under QITE keeps its hash.
    encoder
        For a standard-form T3 ``"stateprep"`` run only: the encoding of :math:`\mathbb{G}`,
        ``"structured"`` or the paper's ``"published"`` one. It
        enters the identity only when it is the published one, so every other record keeps its
        hash.
    tier
        For a ``"stateprep"`` or ``"readout"`` run: ``"T2"``, the exact computation, or ``"T1"``,
        the circuit simulated; a ``"stateprep"`` run may also be
        ``"T3"``, the resource estimate at ``n``.
    shots
        For a T1 ``"readout"`` run only: also sample every measured probability with this many
        shots (``0``: exact probabilities only).
    label
        A free label that groups runs in reports, such as ``"R-D3: exact root only"``. It is
        part of the hash, so two labels never share a record.

    Raises
    ------
    ValueError
        For a faithful variant outside the standard form, the minimax
        filter outside the ``"stateprep"`` stage, the published encoder outside a standard-form
        T3 run, or the chain ablation outside a descriptor ``"build"`` run.
    """

    problem: Problem
    n: tuple[int, ...]
    stage: Stage = "build"
    regime: Regime = "standard"
    mult: MultiplicationMode = "lift"
    truncation: Truncation = "none"
    rescale: Rescale | None = None
    carleman: int | None = None
    layout: FirstOrderLayout = "mass"
    lift: LiftLayout = "symmetric"
    verify: Verify = "exact"
    resources: bool = True
    kind: InitialKind = "geometric"
    geometric_r: tuple[float, ...] = GEOMETRIC_R
    epsilon: float = DEFAULT_EPSILON
    filter: FilterKind = "qite"
    encoder: StandardEncoder = "structured"
    tier: Tier = "T2"
    shots: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        """Refuse a faithful panel outside the standard form, and an option the run cannot use."""
        if self.problem.variant == "faithful" and self.regime != "standard":
            raise ValueError(
                f"the {self.regime} form does not run a panel as the paper printed it: "
                f"{self.problem.key}"
            )
        if self.filter != "qite" and self.stage != "stateprep":
            raise ValueError(f"the {self.stage!r} stage applies QITE only, not {self.filter!r}")
        estimated = self.regime == "standard" and self.stage == "stateprep" and self.tier == "T3"
        if self.encoder != "structured" and not estimated:
            raise ValueError(f"the {self.encoder!r} encoder runs in a standard-form T3 run only")
        if self.layout != "mass" and (self.stage != "build" or self.regime != "descriptor"):
            raise ValueError("the chain layout is a classical ablation of the descriptor")
        if self.lift != "symmetric" and self.stage != "lift":
            raise ValueError("a lift's layout is chosen in the 'lift' stage only")
        if (self.verify == "ir") != (self.stage == "lift") and self.verify in ("ir", "probe"):
            raise ValueError("a 'lift' run is checked exactly or by IR, and only it by IR")
        if self.stage == "lift" and (
            self.carleman is None or self.n != (0,) or self.regime != "standard"
        ):
            raise ValueError(
                "a lift run encodes a Carleman order's coupling alone: n = (0,), the default form"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the specification as strict-JSON data (the record's ``spec``)."""
        rescaling: dict[str, JSONValue] = (
            {"rescale": self.rescale or "block"} if self.regime == "descriptor" else {}
        )
        filtering: dict[str, JSONValue] = {"filter": self.filter} if self.filter != "qite" else {}
        encoding: dict[str, JSONValue] = (
            {"encoder": self.encoder} if self.encoder != "structured" else {}
        )
        lifting: dict[str, JSONValue] = {} if self.carleman is None else {"carleman": self.carleman}
        chaining: dict[str, JSONValue] = {} if self.layout == "mass" else {"layout": self.layout}
        lifting_layout: dict[str, JSONValue] = {"lift": self.lift} if self.stage == "lift" else {}
        return {
            "schema": RUN_SCHEMA,
            "problem": self.problem.to_dict(),
            "problem_digest": self.problem.digest,
            "n": list(self.n),
            "stage": self.stage,
            "regime": self.regime,
            "mult": self.mult,
            "verify": self.verify,
            "resources": self.resources,
            "truncation": self.truncation,
            **rescaling,
            **lifting,
            **chaining,
            **lifting_layout,
            "kind": self.kind,
            "geometric_r": list(self.geometric_r),
            "epsilon": self.epsilon,
            **filtering,
            **encoding,
            "tier": self.tier,
            "shots": self.shots,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> RunSpec:
        """Rebuild a specification from :meth:`to_dict`'s data, such as a record's ``spec``.

        An option :meth:`to_dict` leaves out takes its default, as it had when the data were
        written, so the specification's :attr:`config_hash` is the data's.

        Examples
        --------
        >>> from pihm.problems.ladder import problem
        >>> spec = RunSpec(problem=problem("R2c"), n=(3,), regime="descriptor")
        >>> RunSpec.from_dict(spec.to_dict()).config_hash == spec.config_hash
        True
        """
        given = {item.name for item in dataclasses.fields(cls)} - {"problem", "n", "geometric_r"}
        options = {name: data[name] for name in given if name in data}
        return cls(
            problem=Problem.from_dict(dict(data["problem"])),
            n=tuple(int(k) for k in data["n"]),
            geometric_r=tuple(float(r) for r in data.get("geometric_r", GEOMETRIC_R)),
            **options,
        )

    @cached_property
    def config_hash(self) -> str:
        """The record identity, :func:`pihm.pipeline.records.config_hash` of :meth:`to_dict`."""
        return config_hash(self.to_dict())


def record_directory(root: Path, spec: RunSpec) -> Path:
    """Return where the record of ``spec`` lives under ``root``.

    The layout is ``<root>/<stage>/<problem>/<regime>-n<n>-<hash>``, the problem written as its
    :attr:`~pihm.problems.spec.Problem.key` (``R2a``, or ``R2a/faithful`` for the panel as the
    paper printed it) and ``n`` per axis (``n3`` or ``n4x4``), so that a listing reads naturally.
    """
    size = "x".join(str(k) for k in spec.n)
    leaf = f"{spec.regime}-n{size}-{spec.config_hash}"
    return root / spec.stage / spec.problem.key / leaf
