"""The build stage, campaigns and the summary table."""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from pihm.assemble import assemble
from pihm.classical import physical_reference
from pihm.pipeline.campaign import (
    Allocation,
    classes,
    expand,
    load_campaign,
    modify,
    prune,
    run_campaign,
)
from pihm.pipeline.records import read_record
from pihm.pipeline.spec import RunSpec, record_directory
from pihm.pipeline.stages import run
from pihm.pipeline.summary import (
    IDENTITY,
    cases,
    collect,
    encoding_comparison,
    filter_prose,
    keep_best_candidate,
)
from pihm.problems import ladder


def _spec(**changes: object) -> RunSpec:
    base = RunSpec(problem=ladder.problem("R2a"), n=(3,))
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def _descriptor(**changes: object) -> RunSpec:
    return _spec(regime="descriptor", **changes)


@pytest.mark.parametrize(
    "spec",
    [
        RunSpec(problem=ladder.problem("R2c", "faithful"), n=(3,), truncation="tau"),
        RunSpec(problem=ladder.problem("R5b"), n=(2, 2), regime="descriptor"),
        RunSpec(
            problem=ladder.problem("R8"), n=(3,), regime="descriptor", rescale="none", carleman=3
        ),
        RunSpec(problem=ladder.problem("R7"), n=(3,), regime="descriptor", layout="chain"),
        RunSpec(
            problem=ladder.problem("R1"),
            n=(4,),
            stage="stateprep",
            tier="T3",
            filter="minimax",
            encoder="published",
            geometric_r=(0.5,),
            label="per candidate",
        ),
        RunSpec(
            problem=ladder.problem("R9c-M4"),
            n=(0,),
            stage="lift",
            carleman=2,
            lift="tensor",
            verify="ir",
        ),
    ],
)
def test_a_spec_is_rebuilt_from_its_record_data(spec: RunSpec) -> None:
    rebuilt = RunSpec.from_dict(spec.to_dict())
    assert rebuilt.config_hash == spec.config_hash
    assert rebuilt.to_dict() == spec.to_dict()


def test_the_spec_names_everything_the_run_depends_on() -> None:
    spec = _spec()
    data = spec.to_dict()
    assert data["problem_digest"] == ladder.problem("R2a").digest
    assert data["n"] == [3]
    assert (data["regime"], data["mult"], data["truncation"]) == ("standard", "lift", "none")
    hashes = {
        spec.config_hash,
        _spec(label="other").config_hash,
        _spec(truncation="tau").config_hash,
        _spec(n=(4,)).config_hash,
        _spec(problem=ladder.problem("R2a", "faithful")).config_hash,
    }
    assert len(hashes) == 5


def test_record_layout(tmp_path: Path) -> None:
    spec = _spec(n=(3,))
    assert record_directory(tmp_path, spec).relative_to(tmp_path).parts == (
        "build",
        "R2a",
        f"standard-n3-{spec.config_hash}",
    )
    printed = _spec(problem=ladder.problem("R2a", "faithful"))
    assert record_directory(tmp_path, printed).relative_to(tmp_path).parts[:3] == (
        "build",
        "R2a",
        "faithful",
    )
    assert "n4x4" in record_directory(tmp_path, RunSpec(ladder.problem("R5a"), (4, 4))).name


def test_a_build_run_writes_a_complete_record(tmp_path: Path) -> None:
    spec = _spec()
    record = run(spec, tmp_path)
    assert record.status == "ok"
    again = read_record(record_directory(tmp_path, spec))
    assert again.config_hash == spec.config_hash
    metrics = again.metrics
    assert metrics["problem"] == "R2a"
    assert metrics["figure"] == "3b"
    assert metrics["eta_printed"] == 1.29
    assert metrics["eta_ideal"] == pytest.approx(1.29009, abs=1e-5)
    assert metrics["trusted"] is True  # a gap ratio of 69, above ordinary spacing
    assert metrics["kernel_dimension"] == 1
    for key in ("field_err_linf", "field_err_l2", "lambda0", "gap", "readout_amplification"):
        assert math.isfinite(metrics[key])
    assert set(again.timings) >= {"assemble", "svd", "readout", "total"}
    arrays = again.arrays
    assert arrays["state"].shape == (8,)
    assert arrays["field"].shape == arrays["grid_0"].shape == (1001,)
    assert "run R2a stage=build" in again.log


def test_a_failing_run_is_recorded_not_raised(tmp_path: Path) -> None:
    spec = _descriptor(stage="readout")
    record = run(spec, tmp_path)
    assert record.status == "failed"
    assert record.error is not None
    assert "'readout' stage is not built for the descriptor regime" in record.error
    assert read_record(record_directory(tmp_path, spec)).status == "failed"


def test_only_a_descriptor_run_carries_its_block_rescaling_in_its_identity() -> None:
    assert "rescale" not in _spec().to_dict()
    default = _descriptor()
    assert default.to_dict()["rescale"] == "block"
    assert default.config_hash == _descriptor(rescale="block").config_hash
    assert default.config_hash != _descriptor(rescale="none").config_hash


def test_the_faithful_variants_run_in_the_standard_form_only_and_when_listed() -> None:
    """The paper's panels as printed serve the reproduction only."""
    with pytest.raises(ValueError, match="faithful"):
        _spec(problem=ladder.problem("R2a", "faithful"), regime="descriptor")
    grid = {"problems": ["R3", "R4"], "n": [3]}
    for regime in ("standard", "descriptor"):
        runs = expand([{**grid, "regime": regime}])
        assert {s.problem.variant for s in runs} == {"corrected"}
        assert len(runs) == 8
    listed = expand([{**grid, "variants": ["faithful", "corrected"]}])
    assert {s.problem.variant for s in listed} == {"faithful", "corrected"}
    with pytest.raises(ValueError, match="faithful"):
        expand([{**grid, "regime": "descriptor", "variants": ["faithful"]}])
    with pytest.raises(ValueError, match="unknown regime"):
        expand([{**grid, "regime": "spectral"}])


def test_a_descriptor_build_run_records_its_blocks(tmp_path: Path) -> None:
    standard = run(_spec(), tmp_path).metrics
    descriptor = run(_descriptor(), tmp_path).metrics
    assert (standard["blocks"], standard["padding_blocks"], standard["field_qubits"]) == (1, 0, 0)
    assert standard["field_weight"] == 1.0
    assert (descriptor["blocks"], descriptor["padding_blocks"]) == (4, 1)
    assert (descriptor["field_qubits"], descriptor["dimension"]) == (2, 32)
    assert 0.0 < descriptor["field_weight"] < 1.0
    assert descriptor["regime"] == "descriptor"
    assert (descriptor["rescale"], descriptor["scales"]) == ("block", [4.0])
    assert (standard["rescale"], standard["scales"]) == ("none", [1.0])


def test_expansion() -> None:
    specs = expand([{"problems": ["R1", "R2a"], "n": {"from": 2, "to": 3}}])
    assert [(s.problem.key, s.n) for s in specs] == [
        ("R1", (2,)),
        ("R1", (3,)),
        ("R2a", (2,)),
        ("R2a", (3,)),
    ]
    both = expand([{"problems": ["R1", "R2a"], "variants": ["faithful", "corrected"], "n": [2]}])
    assert [s.problem.key for s in both] == ["R1", "R2a/faithful", "R2a"]
    rung = expand([{"problems": ["R5"], "variants": ["faithful"], "n": [3], "label": "x"}])
    assert [s.problem.name for s in rung] == ["R5a", "R5b", "R5c"]
    assert all(s.n == (3, 3) and s.label == "x" for s in rung)
    single = expand([{"problems": ["R1"], "n": [2], "stage": "stateprep", "geometric_r": [-0.5]}])
    assert [s.geometric_r for s in single] == [(-0.5,)]
    with pytest.raises(ValueError, match="unknown campaign keys"):
        expand([{"problems": ["R1"], "n": [2], "colour": "red"}])
    with pytest.raises(ValueError, match="list or"):
        expand([{"problems": ["R1"], "n": "2-3"}])


def test_a_carleman_sweep_expands_with_every_resolution() -> None:
    specs = expand([{"problems": ["R8"], "carleman": {"from": 2, "to": 3}, "n": [3, 4]}])
    assert [(s.carleman, s.n) for s in specs] == [(2, (3,)), (2, (4,)), (3, (3,)), (3, (4,))]
    assert expand([{"problems": ["R8"], "carleman": 4, "n": [3]}])[0].carleman == 4
    with pytest.raises(ValueError, match="Carleman orders"):
        expand([{"problems": ["R8"], "carleman": [2, 3], "n": [5]}])
    resources = expand([{"problems": ["R8"], "carleman": [2, 3], "n": [5], "stage": "encode"}])
    assert [s.carleman for s in resources] == [2, 3]


def test_the_carleman_order_enters_the_identity_only_when_given() -> None:
    problem = ladder.problem("R8")
    lifted = RunSpec(problem, (3,), carleman=2)
    assert lifted.to_dict()["carleman"] == 2
    assert lifted.config_hash != RunSpec(problem, (3,), carleman=3).config_hash
    assert "carleman" not in RunSpec(ladder.problem("R1"), (3,)).to_dict()


def test_the_chain_layout_enters_the_identity_only_as_the_build_ablation() -> None:
    problem = ladder.problem("R1")
    chain = RunSpec(problem, (3,), regime="descriptor", layout="chain")
    assert chain.to_dict()["layout"] == "chain"
    assert "layout" not in RunSpec(problem, (3,), regime="descriptor").to_dict()
    with pytest.raises(ValueError, match="classical ablation"):
        RunSpec(problem, (3,), regime="descriptor", layout="chain", stage="stateprep")


def test_modifications() -> None:
    faithful = ladder.problem("R4d", "faithful")
    rooted = modify(faithful, {"constraints": "corrected"})
    assert rooted.constraints == ladder.problem("R4d", "corrected").constraints
    assert rooted.variant == "faithful"
    chebyshev = modify(faithful, {"source": "chebyshev"})
    source = chebyshev.equations[0].terms[-1].source
    assert source is not None
    assert (source.representation, source.degree) == ("chebyshev", None)
    eleven = modify(faithful, {"source": "maclaurin:11"})
    assert eleven.equations[0].terms[-1].source.degree == 11  # type: ignore[union-attr]
    assert modify(faithful, {"regular": [0.5]}).regular.at == (0.5,)
    assert len({faithful.digest, rooted.digest, chebyshev.digest, eleven.digest}) == 4
    with pytest.raises(ValueError, match="unknown modifications"):
        modify(faithful, {"n": 3})
    with pytest.raises(ValueError, match="representation"):
        modify(faithful, {"source": "taylor:3"})


def test_campaigns_skip_finished_runs_and_collect(tmp_path: Path) -> None:
    campaign_file = tmp_path / "c.yaml"
    campaign_file.write_text("name: c\ngroups:\n  - problems: [R2a]\n    n: [2, 3]\n")
    campaign = load_campaign(campaign_file)
    assert (campaign.name, len(campaign.specs)) == ("c", 2)
    root = tmp_path / "results"
    first = run_campaign(campaign.specs, root)
    stamps = [(d / "meta.json").stat().st_mtime_ns for d in first]
    second = run_campaign(campaign.specs, root)
    assert [(d / "meta.json").stat().st_mtime_ns for d in second] == stamps
    table = collect(root)
    with table.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert list(rows[0])[: len(IDENTITY)] == list(IDENTITY)
    assert {row["status"] for row in rows} == {"ok"}
    assert rows[0]["qubits"] in ("2", "3")
    assert "nan" in {row["eta_printed"] for row in rows} or all(
        float(row["eta_printed"]) == 1.29 for row in rows
    )
    (tmp_path / "bad.yaml").write_text("name: x\n")
    with pytest.raises(ValueError, match="needs a 'groups' list"):
        load_campaign(tmp_path / "bad.yaml")


def test_a_groups_allocation_is_its_runs_and_never_their_identity(tmp_path: Path) -> None:
    campaign_file = tmp_path / "c.yaml"
    campaign_file.write_text(
        "name: c\ngroups:\n"
        "  - problems: [R2a]\n    n: [2, 3]\n    memory: 230\n    hours: 24\n"
        "  - problems: [R1]\n    n: [2]\n"
    )
    campaign = load_campaign(campaign_file)
    assert campaign.allocations == (Allocation(230, 24), Allocation(230, 24), Allocation())
    assert classes(campaign) == {Allocation(230, 24): [0, 1], Allocation(): [2]}
    plain = expand([{"problems": ["R2a"], "n": [2, 3]}])
    assert [s.config_hash for s in campaign.specs[:2]] == [s.config_hash for s in plain]


def test_arrays_hold_both_axes_of_a_field(tmp_path: Path) -> None:
    record = run(RunSpec(ladder.problem("R5b"), (3, 3)), tmp_path)
    assert record.status == "ok"
    np.testing.assert_allclose(record.arrays["grid_0"], np.linspace(-1, 1, 101))
    np.testing.assert_allclose(record.arrays["grid_1"], np.linspace(-1, 1, 101))
    assert record.arrays["field"].shape == (101, 101)


def test_prune_removes_only_records_outside_the_campaign(tmp_path: Path) -> None:
    kept = RunSpec(ladder.problem("R2a"), (2,))
    dropped = RunSpec(ladder.problem("R2a"), (3,))
    run_campaign([kept, dropped], tmp_path)
    (tmp_path / "notes.txt").write_text("not a record")
    removed = prune([kept], tmp_path)
    assert removed == [record_directory(tmp_path, dropped)]
    assert (record_directory(tmp_path, kept) / "meta.json").exists()
    assert (tmp_path / "notes.txt").exists()
    assert prune([kept], tmp_path) == []


def test_keep_best_candidate_keeps_the_larger_probability_per_case() -> None:
    """One ok row per case (the larger p); rows that are not ok are all kept, in order."""
    base = {"variant": "corrected", "tier": "T1", "truncation": "none", "epsilon": 1e-6}
    base |= {"label": ""}
    rows = [
        {**base, "problem": "R1", "n": 2, "status": "ok", "success_probability": 0.2, "id": 0},
        {**base, "problem": "R1", "n": 2, "status": "ok", "success_probability": 0.7, "id": 1},
        {**base, "problem": "R1", "n": 3, "status": "infeasible", "id": 2},
        {**base, "problem": "R1", "n": 3, "status": "ok", "success_probability": 0.1, "id": 3},
    ]
    assert [r["id"] for r in keep_best_candidate(rows)] == [1, 2, 3]
    tie = [{**rows[0], "success_probability": 0.5, "geometric_r": r, "id": r} for r in (-0.5, 0.5)]
    assert [r["id"] for r in keep_best_candidate(tie)] == [0.5]  # a single run tries 0.5 first


def test_each_filter_keeps_its_own_candidates_and_is_named() -> None:
    base = {"variant": "corrected", "tier": "T2", "truncation": "none", "epsilon": 1e-6}
    base |= {"label": "", "problem": "R1", "n": 3, "status": "ok"}
    rows = [
        {**base, "success_probability": 0.2, "id": 0},
        {**base, "success_probability": 0.7, "filter": "minimax", "id": 1},
    ]
    assert [r["id"] for r in keep_best_candidate(rows)] == [0, 1]
    assert filter_prose(rows[:1]) == "the QITE filter at the imaginary time the gap sets"
    assert filter_prose(rows[1:]).startswith("the minimax filter at the degree")


def test_the_encoding_comparison_joins_both_forms_by_case() -> None:
    def encode(problem: str, n: int, alpha: float, status: str = "ok") -> dict[str, object]:
        return {
            "problem": problem,
            "variant": "corrected",
            "n": n,
            "status": status,
            "alpha_R": alpha,
            "n_anc_qubits": 10 + n,
            "cx_count": 100 * n,
        }

    def build(problem: str, n: int, norm_squared: float) -> dict[str, object]:
        return {
            "problem": problem,
            "variant": "corrected",
            "n": n,
            "status": "ok",
            "hamiltonian_norm": norm_squared,
            "label": "",
        }

    standard = (
        [encode("R2a", 3, 40.0), encode("R2a", 4, 80.0), encode("R1", 3, 9.0)],
        [build("R2a", 3, 100.0)],
    )
    builds = [build("R2a", 3, 36.0), {**build("R2a", 3, 1.0), "label": "unscaled"}]
    with pytest.raises(ValueError, match="differ in \\['label'\\]"):
        cases(builds)
    descriptor = (
        [encode("R2a", 3, 12.0), encode("R2a", 4, 20.0, "failed")],
        list(cases(builds, label="").values()),
    )
    (row,) = encoding_comparison(standard, descriptor)
    assert (row["problem"], row["n"]) == ("R2a", 3)
    assert (row["looseness_standard"], row["looseness_descriptor"]) == (4.0, 2.0)
    assert (row["ancilla_descriptor"], row["cx_standard"]) == (13, 300)
    no_build = encoding_comparison((standard[0], []), descriptor)
    assert math.isnan(no_build[0]["looseness_standard"])


def test_a_descriptor_stateprep_run_records_its_rescaling(tmp_path: Path) -> None:
    record = run(_descriptor(stage="stateprep"), tmp_path)
    assert record.status == "ok"
    metrics = record.metrics
    assert (metrics["rescale"], metrics["scales"], metrics["regime"]) == (
        "block",
        [4.0],
        "descriptor",
    )
    assert metrics["infidelity"] <= metrics["infidelity_bound"]
    assert 0.0 < metrics["gamma2"] <= 1.0
    assert metrics["field_err_linf"] < 1e-3


@pytest.mark.parametrize(("regime", "n"), [("standard", 2), ("descriptor", 3)])
def test_a_filters_run_places_the_physical_spectrum_on_the_map(
    regime: str, n: int, tmp_path: Path
) -> None:
    # R2a's descriptor state has a padding block, whose singular value 1 lies inside the physical
    # gap at n = 3: a spectrum that kept it would put x_1 there.
    spec = _spec(problem=ladder.problem("R2a", "corrected"), n=(n,), stage="filters", regime=regime)
    record = run(spec, tmp_path)
    assert record.status == "ok"
    metrics = record.metrics
    residual = assemble(spec.problem, spec.n, regime=spec.regime, rescale=spec.rescale)
    s = physical_reference(residual).singular_values
    gap = s[1] ** 2 - s[0] ** 2
    assert metrics["dimension"] == residual.physical_columns.size
    assert metrics["delta"] == pytest.approx(2.0 * gap / metrics["alpha_R"] ** 2)
    assert metrics["offset"] == pytest.approx(s[0] ** 2 / gap)
    assert ("rescale" in metrics) == (regime == "descriptor")
    for name in ("minimax", "qite"):
        assert metrics[f"t2_{name}_infidelity"] <= metrics["epsilon"]
        assert metrics[f"t2_{name}_phases_error"] <= 1e-12


def test_only_a_minimax_run_carries_its_filter_in_its_identity() -> None:
    qite = _spec(stage="stateprep")
    minimax = _spec(stage="stateprep", filter="minimax")
    assert "filter" not in qite.to_dict()
    assert minimax.to_dict()["filter"] == "minimax"
    assert qite.config_hash != minimax.config_hash
    with pytest.raises(ValueError, match="applies QITE only"):
        _spec(stage="protocol", filter="minimax")


def _estimate(**changes: object) -> RunSpec:
    base = {"problem": ladder.problem("R2a", "corrected"), "stage": "stateprep", "tier": "T3"}
    return _spec(**(base | changes))


def test_only_a_published_estimate_carries_its_encoder_in_its_identity() -> None:
    structured, published = _estimate(), _estimate(encoder="published")
    assert "encoder" not in structured.to_dict()
    assert published.to_dict()["encoder"] == "published"
    assert structured.config_hash != published.config_hash
    for changes in ({"tier": "T2"}, {"stage": "encode"}, {"regime": "descriptor"}):
        with pytest.raises(ValueError, match="standard-form T3 run only"):
            _estimate(encoder="published", **changes)


def test_a_published_estimate_costs_the_papers_encoding(tmp_path: Path) -> None:
    published = run(_estimate(encoder="published", n=(2,)), tmp_path).metrics
    structured = run(_estimate(n=(2,)), tmp_path).metrics
    assert published["encoder"] == "published"
    assert "encoder" not in structured
    assert published["gap"] == structured["gap"]
    assert published["alpha_R"] > structured["alpha_R"]
    assert published["degree"] > structured["degree"]


def test_a_minimax_stateprep_run_records_its_filter(tmp_path: Path) -> None:
    record = run(_descriptor(stage="stateprep", filter="minimax"), tmp_path)
    assert record.status == "ok"
    metrics = record.metrics
    assert (metrics["filter"], metrics["tau"]) == ("minimax", None)
    assert metrics["infidelity"] <= metrics["infidelity_bound"]
