"""The ``"lift"`` stage: a Carleman lift's coupling alone, in either layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from pihm.pipeline.catalogue import run_key
from pihm.pipeline.spec import LiftLayout, RunSpec, Verify
from pihm.pipeline.stages import run
from pihm.problems.ladder import problem


def _lift(name: str, order: int, layout: LiftLayout, verify: Verify = "ir") -> RunSpec:
    return RunSpec(problem(name), (0,), stage="lift", carleman=order, lift=layout, verify=verify)


@pytest.mark.parametrize("layout", ["symmetric", "tensor"])
def test_a_lift_run_records_its_layout_verified(layout: LiftLayout, tmp_path: Path) -> None:
    record = run(_lift("R9c-M4", 3, layout), tmp_path)
    assert record.status == "ok", record.error
    metrics = record.metrics
    assert metrics["verify_passed"] is True
    assert metrics["lift"] == layout
    assert metrics["modes"] == 4
    if layout == "tensor":
        assert metrics["terms"] == 5
        assert metrics["components"] == 4 * 2 + (4 * 2) ** 2 + (4 * 2) ** 3
    else:
        assert metrics["components"] == 34
    assert float(metrics["alpha_over_norm"]) >= 1.0


def test_the_symmetric_lift_is_exact_on_its_field_register(tmp_path: Path) -> None:
    record = run(_lift("R9c-M4", 2, "symmetric", "exact"), tmp_path)
    assert record.metrics["verify_passed"] is True
    assert float(record.metrics["verify_max_error"]) < 1e-14


def test_a_lift_run_needs_an_order_no_time_axis_and_the_default_form() -> None:
    with pytest.raises(ValueError, match="coupling alone"):
        RunSpec(problem("R9c-M4"), (0,), stage="lift")
    with pytest.raises(ValueError, match="coupling alone"):
        RunSpec(problem("R9c-M4"), (3,), stage="lift", carleman=2)
    with pytest.raises(ValueError, match="coupling alone"):
        RunSpec(problem("R9c-M4"), (0,), stage="lift", carleman=2, regime="descriptor")
    with pytest.raises(ValueError, match="'lift' stage only"):
        RunSpec(problem("R9c-M4"), (3,), carleman=2, lift="tensor")
    with pytest.raises(ValueError, match="by IR"):
        RunSpec(problem("R9c-M4"), (3,), stage="encode", carleman=2, verify="ir")
    with pytest.raises(ValueError, match="by IR"):
        RunSpec(problem("R9c-M4"), (0,), stage="lift", carleman=2, verify="probe")


def test_the_layout_enters_only_a_lift_runs_identity_and_key() -> None:
    build = RunSpec(problem("R9c-M4"), (3,), carleman=2)
    assert "lift" not in build.to_dict()
    tensor = _lift("R9c-M4", 2, "tensor").to_dict()
    assert tensor["lift"] == "tensor"
    assert run_key(tensor).startswith("lift/R9c-M4/standard/n0/")
    assert "lift=tensor" in run_key(tensor)
    assert "lift=" not in run_key(_lift("R9c-M4", 2, "symmetric").to_dict())
