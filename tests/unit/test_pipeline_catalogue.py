"""The results catalogue (``pihm.pipeline.catalogue``): run keys, and each expected run's status."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from pihm.pipeline.campaign import Campaign, modify
from pihm.pipeline.catalogue import catalogue, label_token, run_key
from pihm.pipeline.records import Provenance, Record, write_record
from pihm.pipeline.spec import RunSpec, record_directory
from pihm.problems.ladder import problem


def _write(
    root: Path,
    spec: RunSpec,
    status: str = "ok",
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    record = Record(
        spec=spec.to_dict(),
        status=status,  # type: ignore[arg-type]
        metrics=metrics or {},
        provenance=Provenance.capture(),
        timings={"total": 1.5},
        error=error,
    )
    write_record(record, record_directory(root, spec))


def _campaign(*specs: RunSpec, name: str = "c") -> Campaign:
    return Campaign(name=name, description="", specs=specs)


def test_keys_read_like_the_record_directory() -> None:
    corrected = RunSpec(problem=problem("R2a"), n=(3,)).to_dict()
    assert run_key(corrected) == "build/R2a/standard/n3"
    named = corrected | {"problem": {**problem("R2a").to_dict(), "variant": "corrected"}}
    assert run_key(named) == "build/R2a/standard/n3"
    faithful = problem("R2a", "faithful")
    build = RunSpec(problem=faithful, n=(3,))
    assert run_key(build.to_dict()) == "build/R2a/faithful/standard/n3"
    t1 = RunSpec(problem=faithful, n=(3,), stage="stateprep", tier="T1", geometric_r=(0.5,))
    assert run_key(t1.to_dict()) == "stateprep/R2a/faithful/standard/n3/T1"
    tau = RunSpec(problem=faithful, n=(5,), truncation="tau", epsilon=1e-8, label="tau ablation")
    assert (
        run_key(tau.to_dict())
        == "build/R2a/faithful/standard/n5/truncation=tau/epsilon=1e-08/tau-ablation"
    )
    assert run_key(tau.to_dict(), {"tau ablation": ""}).endswith("/epsilon=1e-08")
    assert run_key(tau.to_dict(), {"tau ablation": "tau"}).endswith("/tau")
    two_axes = RunSpec(problem=problem("R5a"), n=(3, 3), stage="readout")
    assert run_key(two_axes.to_dict()) == "readout/R5a/standard/n3x3/T2"


def test_a_minimax_run_names_its_filter_and_a_qite_run_keeps_its_key() -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,), stage="stateprep", regime="descriptor")
    assert run_key(spec.to_dict()) == "stateprep/R1/descriptor/n3/T2"
    minimax = dataclasses.replace(spec, filter="minimax")
    assert run_key(minimax.to_dict()) == "stateprep/R1/descriptor/n3/T2/filter=minimax"


def test_a_published_estimate_names_its_encoder() -> None:
    spec = RunSpec(problem=problem("R2a"), n=(3,), stage="stateprep", tier="T3")
    assert run_key(spec.to_dict()) == "stateprep/R2a/standard/n3/T3"
    published = run_key(dataclasses.replace(spec, encoder="published").to_dict())
    assert published == "stateprep/R2a/standard/n3/T3/encoder=published"


def test_an_older_record_without_a_newer_field_keys_as_its_default() -> None:
    spec = RunSpec(problem=problem("R1"), n=(2,)).to_dict()
    del spec["geometric_r"], spec["tier"]
    assert run_key(spec) == "build/R1/standard/n2"


def test_label_tokens() -> None:
    assert label_token("exact root, Maclaurin p = 7") == "exact-root-maclaurin-p-7"
    assert label_token("x_s = 0.5") != label_token("x_s = -0.5")
    spec = RunSpec(problem=problem("R1"), n=(2,), label="odd").to_dict()
    with pytest.raises(ValueError, match="not a key segment"):
        run_key(spec, {"odd": "a/b"})


def test_every_expected_run_has_an_entry(tmp_path: Path) -> None:
    specs = [RunSpec(problem=problem("R1"), n=(n,)) for n in (2, 3, 4, 5)]
    _write(tmp_path, specs[0], metrics={"eta": 0.83, "degree": 89, "trusted": True, "x": "text"})
    reasons = ["memory 300 GiB over the 230 GiB budget"]
    _write(
        tmp_path,
        specs[1],
        "infeasible",
        {"predicted_peak_gib": 300.0, "infeasible_reasons": reasons},
    )
    _write(tmp_path, specs[2], "failed", error="Traceback ...\nValueError: no kernel")
    entries = catalogue([(_campaign(*specs, name="p9"), tmp_path)])

    ok = entries["build/R1/standard/n2"]
    assert (ok.status, ok.reason) == ("ok", "")
    assert ok.values == {"eta": 0.83, "degree": 89, "time_total": 1.5}
    assert ok.flags == {"trusted": True}
    assert ok.sources[0].config_hash == specs[0].config_hash

    infeasible = entries["build/R1/standard/n3"]
    assert infeasible.status == "infeasible"
    assert infeasible.reason == reasons[0]
    assert infeasible.values["predicted_peak_gib"] == 300.0

    failed = entries["build/R1/standard/n4"]
    assert (failed.status, failed.reason, failed.values) == ("failed", "ValueError: no kernel", {})

    pending = entries["build/R1/standard/n5"]
    assert (pending.status, pending.reason) == ("pending", "not yet run (p9)")


def test_a_record_of_an_earlier_configuration_is_stale(tmp_path: Path) -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,))
    earlier = dataclasses.replace(spec, problem=modify(spec.problem, {"regular": [0.5]}))
    _write(tmp_path, earlier, metrics={"eta": 0.9})
    orphan = RunSpec(problem=problem("R1"), n=(4,))
    _write(tmp_path, orphan, metrics={"eta": 1.1})
    entries = catalogue([(_campaign(spec, name="p9"), tmp_path)])

    stale = entries["build/R1/standard/n3"]
    assert stale.status == "stale"
    assert stale.values["eta"] == 0.9
    assert earlier.config_hash in stale.reason
    assert stale.reason.endswith("rerun p9")

    gone = entries["build/R1/standard/n4"]
    assert (gone.status, gone.reason) == ("stale", "no campaign expects this run any more")


def test_initial_state_candidates_share_a_key_and_the_best_is_quoted(tmp_path: Path) -> None:
    base = RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", tier="T1")
    plus, minus = (dataclasses.replace(base, geometric_r=(r,)) for r in (0.5, -0.5))
    campaign = _campaign(plus, minus, name="p9")
    key = "stateprep/R1/standard/n2/T1"

    _write(tmp_path, plus, metrics={"success_probability": 0.2, "geometric_r": 0.5})
    partial = catalogue([(campaign, tmp_path)])[key]
    assert (partial.status, partial.reason) == (
        "pending",
        "1 of 2 candidate runs recorded; not yet run (p9)",
    )

    _write(tmp_path, minus, metrics={"success_probability": 0.7, "geometric_r": -0.5})
    entry = catalogue([(campaign, tmp_path)])[key]
    assert entry.status == "ok"
    assert entry.values["success_probability"] == 0.7
    assert entry.sources[0].config_hash == minus.config_hash


def test_runs_that_differ_beyond_their_candidates_may_not_share_a_key() -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,))
    moved = dataclasses.replace(spec, problem=modify(spec.problem, {"regular": [0.5]}))
    with pytest.raises(ValueError, match="share the key build/R1/standard/n3"):
        catalogue([(_campaign(spec, moved), Path("missing"))])
