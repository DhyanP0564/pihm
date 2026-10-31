"""Every rung through the build stage at its smallest resolution.

The integration tier runs each rung end to end: problem, assembly, SVD reference, readout,
metrics and record.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pihm.pipeline.records import dumps_strict, read_record
from pihm.pipeline.spec import RunSpec, record_directory
from pihm.pipeline.stages import run
from pihm.problems import ladder
from pihm.problems.spec import Variant

#: The smallest resolution of each rung's classical sweep.
SMALLEST = {
    "R1": 2,
    "R2": 2,
    "R3": 3,
    "R4": 3,
    "R5": 2,
    "R6": 2,
    "R7": 2,
    "R8": 3,
    "R9": 3,
    "R10": 3,
}

#: The smallest Carleman order of a rung that is lifted.
CARLEMAN = {"R8": 2, "R9": 2, "R10": 2}

#: R10's truncations past twenty-eight modes are campaigns': their smallest build takes
#: minutes, and the field scale's no build reaches.
CASES = [
    (name, variant)
    for name in ladder.names()
    for variant in ("faithful", "corrected")
    if (variant == "corrected" or name in ladder.PANELS)
    and (ladder.rung(name) != "R10" or len(ladder.problem(name).fields) <= 28)
]


@pytest.mark.parametrize(("name", "variant"), CASES)
@pytest.mark.nightly
def test_rung_runs_end_to_end(name: str, variant: Variant, tmp_path: Path) -> None:
    problem = ladder.problem(name, variant)
    rung = ladder.rung(name)
    n = SMALLEST[rung]
    spec = RunSpec(problem, (n,) * len(problem.fields[0].axes), carleman=CARLEMAN.get(rung))
    record = run(spec, tmp_path)
    assert record.status == "ok", record.error
    stored = read_record(record_directory(tmp_path, spec))
    # Strict JSON compares NaN (a non-panel's printed eta) as a tagged value, not by identity.
    assert dumps_strict(stored.metrics) == dumps_strict(record.metrics)
    for key in ("eta", "eta_ideal", "field_err_linf", "infidelity", "lambda1", "hamiltonian_norm"):
        assert math.isfinite(stored.metrics[key]), key
    assert stored.metrics["selection"] == ("product" if name.startswith("R6") else "svd")
