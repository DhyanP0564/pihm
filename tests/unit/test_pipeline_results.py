"""Checks of the results found by campaign name and the shared selections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pihm.pipeline.records import source_digest
from pihm.pipeline.results import (
    CONFIG,
    Results,
    cost_comparison,
    degree_exponents,
    encoding_scaling,
    primary_preparations,
    reference_builds,
)
from pihm.pipeline.summary import read_summary

ROOT = Path(__file__).resolve().parents[2]


def _results(root: Path) -> Results:
    """Return the results under ``root``: campaign ``a`` collected, ``b`` not yet."""
    (root / "examples").mkdir()
    (root / CONFIG).write_text("campaigns:\n  a: results/a\n  b: results/b\n")
    (root / "results" / "a").mkdir(parents=True)
    (root / "results" / "a" / "summary.csv").write_text(
        "config_hash,status,problem\nh1,ok,R1\nh2,failed,R2a\n"
    )
    return Results.at(root)


def test_a_campaign_is_found_by_its_name(tmp_path: Path) -> None:
    results = _results(tmp_path)
    assert results.directory("a") == (tmp_path / "results" / "a").resolve()
    assert [row["problem"] for row in results.rows("a")] == ["R1", "R2a"]
    assert [row["config_hash"] for row in results.rows("a", status="ok")] == ["h1"]
    with pytest.raises(KeyError, match="not a campaign"):
        results.directory("c")


def test_the_accepted_sources_are_the_trees_and_the_configured_ones(tmp_path: Path) -> None:
    results = _results(tmp_path)
    assert results.accepted_sources() == {source_digest()}
    with (tmp_path / CONFIG).open("a") as config:
        config.write("sources:\n  0123456789abcdef: an earlier source, and why it stands\n")
    assert Results.at(tmp_path).accepted_sources() == {source_digest(), "0123456789abcdef"}


def test_a_campaign_without_a_summary_is_missing_and_described_so(tmp_path: Path) -> None:
    results = _results(tmp_path)
    assert results.missing(["a", "b"]) == ("b",)
    lines = results.describe(["a", "b"]).splitlines()
    assert lines[0] == "a: 2 records, 1 ok"
    assert lines[1].startswith("b: no results in ")


def test_the_repository_names_every_campaign_file() -> None:
    campaigns = {path.stem for path in (ROOT / "examples" / "campaigns").glob("*.yaml")}
    named = set(Results.at(ROOT).directories)
    assert named <= campaigns


def _preparation(**columns: Any) -> dict[str, Any]:
    """Return a primary T2 row of R1 at n = 3, with ``columns`` changed."""
    row = {
        "status": "ok",
        "tier": "T2",
        "initial_kind": "geometric",
        "truncation": "none",
        "label": "",
        "problem": "R1",
        "variant": "corrected",
        "n": 3,
    }
    return row | columns


def test_a_primary_preparation_is_the_geometric_untruncated_t2_run() -> None:
    primary = [
        _preparation(),
        _preparation(label="paper's own construction", problem="R2a"),
    ]
    others = [
        _preparation(status="failed"),
        _preparation(tier="T1"),
        _preparation(initial_kind="uniform", label="initial-state control"),
        _preparation(label="per candidate"),
        _preparation(label="doubled space", problem="R6a"),
        _preparation(truncation="tau"),
    ]
    assert primary_preparations(primary + others) == primary
    assert primary_preparations(primary, problem="R2a") == primary[1:]


def test_the_reference_build_is_the_unlabelled_lifted_assembly() -> None:
    build = {"status": "ok", "problem": "R1", "variant": "corrected", "n": 3, "mult": "lift"}
    rows = [
        build | {"label": "", "truncation": "none"},
        build | {"label": "unscaled", "truncation": "none"},
        build | {"label": "", "truncation": "tau"},
    ]
    assert list(reference_builds(rows).values()) == rows[:1]


def test_the_cost_comparison_is_the_standard_forms_cost_over_the_descriptors() -> None:
    case = {"status": "ok", "problem": "R2a", "variant": "corrected", "n": 4}
    standard = [case | {"gate_count": 6.0e6, "preparations": 3.0, "degree": 1000}]
    descriptor = [case | {"gate_count": 1.0e5, "preparations": 1.0, "degree": 50}]
    build = case | {"label": "", "truncation": "none", "mult": "lift", "trusted": True}
    (row,) = cost_comparison((standard, [build]), (descriptor, [build | {"trusted": False}]))
    assert row["total"] == pytest.approx(60.0)
    assert row["preparations"] == pytest.approx(3.0)
    assert row["degree"] == pytest.approx(20.0)
    assert row["step"] == pytest.approx(row["total"] / (row["preparations"] * row["degree"]))
    assert row["trusted"] is False
    assert np.isnan(row["ideal_degree"])  # such records have no ideal degree
    (unbuilt,) = cost_comparison((standard, [build]), (descriptor, []))
    assert unbuilt["trusted"] is False
    (ideal,) = cost_comparison(
        ([standard[0] | {"ideal_degree": 250}], [build]),
        ([descriptor[0] | {"ideal_degree": 25}], [build]),
    )
    assert ideal["ideal_degree"] == pytest.approx(10.0)


def _estimate(n: int, alpha: float, degree: float) -> dict[str, Any]:
    """Return a T3 row of R2a at ``n`` with ``alpha`` and ``degree``."""
    case = {"status": "ok", "problem": "R2a", "variant": "corrected", "n": n}
    return case | {"alpha_R": alpha, "gap": 7.0, "degree": degree, "gate_count": 10.0 * degree}


def _build(n: int, trusted: bool = True) -> dict[str, Any]:
    """Return a reference build of R2a at ``n`` whose Hamiltonian's norm is 16**n."""
    case = {"status": "ok", "problem": "R2a", "variant": "corrected", "n": n}
    lifted = {"label": "", "truncation": "none", "mult": "lift"}
    return case | lifted | {"hamiltonian_norm": 16.0**n, "trusted": trusted}


def test_the_encoding_scaling_holds_the_two_ratios_behind_the_degree() -> None:
    estimates = [_estimate(n, 4.0**n, 8.0**n) for n in (2, 3, 4)]
    builds = [_build(2, trusted=False), _build(3), _build(4)]
    measured = [{"status": "ok", "problem": "R2a", "variant": "corrected", "n": 3, "degree": 500}]
    table = encoding_scaling({"structured": (estimates, builds, measured)})
    assert [row["n"] for row in table] == [2, 3, 4]
    assert all(row["looseness"] == pytest.approx(1.0) for row in table)  # alpha_R² = 16**n
    assert table[1]["relative_gap"] == pytest.approx(7.0 / 16.0**3)
    assert [row["measured_degree"] for row in table] == [None, 500.0, None]
    assert [row["trusted"] for row in table] == [False, True, True]
    assert encoding_scaling({"structured": (estimates, builds[1:], [])})[0]["n"] == 3


def test_the_degree_exponent_is_its_power_law_in_n() -> None:
    estimates = [_estimate(n, 4.0**n, 8.0**n) for n in (2, 3, 4)]
    table = encoding_scaling({"structured": (estimates, [_build(n) for n in (2, 3, 4)], [])})
    (fit,) = degree_exponents(table, (3, 4))
    assert fit["exponent"] == pytest.approx(3.0)  # 8**n = N**3
    assert fit["ns"] == [3, 4]
    assert degree_exponents(table, (4,)) == []
    (cx,) = degree_exponents(table, (2, 3, 4), "cx")
    assert cx["exponent"] == pytest.approx(3.0)


def test_a_hash_that_reads_as_a_number_stays_text(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text("config_hash,status,problem,n,degree\n596353258303e911,ok,R2c,2,544\n")
    (row,) = read_summary(summary)
    assert row["config_hash"] == "596353258303e911"
    assert (row["n"], row["degree"]) == (2, 544)
