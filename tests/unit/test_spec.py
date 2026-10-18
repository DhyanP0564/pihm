"""Problem specifications: validation, serialisation and digests."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
import yaml

from pihm.problems import ladder
from pihm.problems.spec import (
    Axis,
    Constraint,
    Datum,
    Equation,
    Factor,
    Field,
    Polynomial,
    Problem,
    Reference,
    Regular,
    Root,
    Source,
    Term,
)

#: Every problem, then every panel of the paper as printed.
ALL = [(name, "corrected") for name in ladder.names()] + [
    (name, "faithful") for name in ladder.PANELS
]


def _base() -> Problem:
    return ladder.problem("R2a", "faithful")


@pytest.mark.parametrize(("name", "variant"), ALL)
def test_yaml_round_trip_and_digest(name: str, variant: str) -> None:
    problem = ladder.problem(name, variant)
    text = problem.to_yaml()
    again = Problem.from_yaml(text)
    assert again == problem
    assert hash(again) == hash(problem)
    assert again.digest == problem.digest
    assert yaml.safe_load(text) == problem.to_dict()


def test_digests_are_distinct_and_track_every_value() -> None:
    digests = {ladder.problem(n, v).digest for n, v in ALL}
    assert len(digests) == len(ALL)
    base = _base()
    moved = dataclasses.replace(base, constraints=(Constraint("zero", "f", "x", -0.999),))
    assert moved.digest != base.digest
    assert dataclasses.replace(base, description="edited").digest != base.digest


def test_defaults_are_left_out_of_the_plain_form() -> None:
    """New optional fields must not change the digests of existing problems."""
    plain = _base().to_dict()
    assert "label" not in plain["axes"][0]
    assert "weight" not in plain["fields"][0]
    assert "derivative" not in plain["constraints"][0]
    assert "variant" not in ladder.problem("R2a").to_dict()


def test_properties() -> None:
    assert (_base().order, _base().degree, _base().is_linear) == (2, 1, True)
    assert ladder.problem("R6a").degree == 2
    assert not ladder.problem("R6a").is_linear
    assert ladder.problem("R5c").order == 2
    assert _base().key == "R2a/faithful"
    assert ladder.problem("R2a").key == "R2a"
    assert Polynomial("x", (1.0, 0.0, 0.0)).is_constant
    assert Polynomial("x", (0.0, 2.0, 0.0)).degree == 1


def _replace(**changes: Any) -> Problem:
    return dataclasses.replace(_base(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"variant": "other"}, "variant"),
        ({"axes": (Axis("x"), Axis("x"))}, "distinct"),
        ({"fields": (Field("f", ("y",)),)}, "known axes"),
        ({"equations": ()}, "one equation per field"),
        (
            {"equations": (Equation((Term(1.0, (Factor("g", (1,)),)),)),)},
            "unknown field",
        ),
        (
            {"equations": (Equation((Term(1.0, (Factor("f", (1, 1)),)),)),)},
            "one derivative order per axis",
        ),
        (
            {"equations": (Equation((Term(1.0, (), Polynomial("y", (1.0,))),)),)},
            "multiplier axis",
        ),
        ({"constraints": (Constraint("zero", "f", "y", 0.0),)}, "not an axis"),
        ({"data": (Datum("f", "x", 0.0, "y + 1"),)}, "allowed name"),
        ({"regular": Regular("f", (0.0, 0.0))}, "one coordinate per axis"),
        ({"reference": Reference("expression", "exp(-y)")}, "allowed name"),
    ],
)
def test_validation(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _replace(**changes)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Axis("pi"), "cannot name an axis"),
        (lambda: Polynomial("x", ()), "at least one"),
        (lambda: Source("exp(x)", "maclaurin"), "needs a degree"),
        (lambda: Source("exp(x)", "chebyshev", 3), "needs a degree"),
        (lambda: Source("exp(x)", "maclaurin", -1), "non-negative"),
        (lambda: Factor("f", (-1,)), "non-negative"),
        (lambda: Term(1.0, (Factor("f", (0,)),), None, Source("x")), "cannot have factors"),
        (lambda: Term(1.0, (), Polynomial("x", (1.0,)), Source("x")), "not both"),
        (lambda: Equation(()), "at least one term"),
        (lambda: Constraint("zero", "f", "x", 1.5), "lie in"),
        (lambda: Constraint("datum", "f", "x", 0.0), "needs a datum"),
        (lambda: Constraint("zero", "f", "x", 0.0, datum="1"), "needs a datum"),
        (lambda: Constraint("datum", "f", "x", 0.0, 1, "1"), "impose values"),
        (lambda: Constraint("collapse", "f", "x", 0.0), "unknown constraint kind"),  # type: ignore[arg-type]
        (lambda: Datum("f", "x", 0.0, "1", -1), "non-negative"),
        (lambda: Regular("f", (2.0,)), "lie in"),
        (lambda: Reference("expression"), "needs an expression"),
        (lambda: Reference("bvp", "x"), "needs an expression"),
        (lambda: Reference("expression", "x", ("x", "x")), "one per field"),
        (lambda: Constraint("ratio", "f", "x", 0.0, 1), "not their slopes"),
    ],
)
def test_component_validation(build: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_bvp_reference_needs_one_axis() -> None:
    with pytest.raises(ValueError, match="one field on one axis"):
        dataclasses.replace(ladder.problem("R5a"), reference=Reference("bvp"))


def test_from_dict_is_strict() -> None:
    plain = _base().to_dict()
    with pytest.raises(ValueError, match="unknown keys"):
        Problem.from_dict({**plain, "colour": "red"})
    broken = {**plain, "regular": {"field": "f", "at": ["zero"]}}
    with pytest.raises(ValueError, match="number"):
        Problem.from_dict(broken)
    with pytest.raises(ValueError, match="mapping"):
        Problem.from_yaml("- a list")


def test_a_root_defined_constraint_is_identified_by_its_recipe() -> None:
    # The corrected variants' roots are found numerically, and their last bits depend on the
    # machine; the digest holds the recipe only, and the problem resolves the root when built.
    problem = ladder.problem("R4a", "corrected")
    (constraint,) = problem.constraints
    assert constraint.root is not None
    assert "at" not in problem.to_dict()["constraints"][0]
    nudged = dataclasses.replace(constraint, at=math.nextafter(constraint.at, 1.0))
    assert dataclasses.replace(problem, constraints=(nudged,)).digest == problem.digest
    elsewhere = dataclasses.replace(constraint, at=math.nan, root=Root(0.0))
    with pytest.raises(ValueError, match="root"):
        dataclasses.replace(problem, constraints=(elsewhere,))
    rebuilt = Problem.from_dict(problem.to_dict())
    assert rebuilt.constraints[0].at == constraint.at


def test_a_root_recipe_changes_the_digest_when_it_changes() -> None:
    problem = ladder.problem("R4a", "corrected")
    (constraint,) = problem.constraints
    assert constraint.root is not None
    moved = dataclasses.replace(constraint, at=math.nan, root=Root(-0.19))
    assert dataclasses.replace(problem, constraints=(moved,)).digest != problem.digest


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fields": (Field("u", ("x",)), Field("v", ("x",), "1 - x**2"))}, "no readout weight"),
        ({"reference": Reference("expression", expressions=("sin(x)",))}, "one closed form"),
        ({"data": (Datum("v", "x", -1.0, "pi"),)}, "needs a datum of 'u'"),
    ],
)
def test_a_system_is_validated(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(ladder.problem("R7"), **change)


def test_a_system_shares_its_axes() -> None:
    problem = ladder.problem("R7")
    with pytest.raises(ValueError, match="share their axes"):
        dataclasses.replace(
            problem,
            axes=(Axis("x"), Axis("y")),
            fields=(Field("u", ("x",)), Field("v", ("y",))),
            constraints=(),
            data=(),
            regular=Regular("u", (0.5,)),
        )


def test_one_field_takes_one_reference_expression() -> None:
    with pytest.raises(ValueError, match="one reference expression"):
        dataclasses.replace(_base(), reference=Reference("expression", expressions=("x", "x**2")))
