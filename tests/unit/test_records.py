"""Record schema v1: strict JSON, configuration hashes and on-disk round trips."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pihm.pipeline.records import (
    SCHEMA_VERSION,
    Provenance,
    Record,
    config_hash,
    dumps_strict,
    loads_strict,
    read_record,
    write_record,
)


def _record(**overrides: object) -> Record:
    fields: dict[str, object] = {
        "spec": {"rung": "R2a", "variant": "corrected", "regime": "descriptor", "n": 5},
        "status": "ok",
        "metrics": {"infidelity": 3.1e-8, "degree": 912, "gap": math.inf, "gamma": math.nan},
        "provenance": Provenance.capture(),
        "timings": {"build": 0.25},
        "arrays": {"state": np.linspace(0.0, 1.0, 7)},
        "log": "built\n",
    }
    fields.update(overrides)
    return Record(**fields)  # type: ignore[arg-type]


@pytest.mark.gotcha
def test_nonfinite_floats_never_appear_as_bare_tokens() -> None:
    """Python's json writes NaN/Infinity by default; the strict writer never does."""
    text = dumps_strict({"a": math.nan, "b": [math.inf, -math.inf], "c": np.float64("nan")})
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text, parse_constant=lambda token: pytest.fail(f"bare {token}"))
    back = loads_strict(text)
    assert math.isnan(back["a"])
    assert back["b"] == [math.inf, -math.inf]
    assert math.isnan(back["c"])


def test_loads_strict_rejects_bare_constants() -> None:
    with pytest.raises(ValueError, match="non-standard"):
        loads_strict('{"x": NaN}')


@pytest.mark.parametrize("bad", [np.zeros(3), 1 + 2j, {1: "int key"}, object()])
def test_dumps_strict_rejects_non_json_types(bad: object) -> None:
    with pytest.raises(TypeError):
        dumps_strict({"x": bad})


def test_reserved_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="reserved"):
        dumps_strict({"$float": "nan"})


def test_numpy_scalars_become_python_values() -> None:
    text = dumps_strict({"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True)})
    assert loads_strict(text) == {"i": 3, "f": 0.5, "b": True}


def test_config_hash_is_order_independent_and_value_sensitive() -> None:
    spec = {"n": 3, "regime": "standard", "options": {"eps": 1e-6, "tier": "emulate"}}
    reordered = {"options": {"tier": "emulate", "eps": 1e-6}, "regime": "standard", "n": 3}
    assert config_hash(spec) == config_hash(reordered)
    assert config_hash(spec) != config_hash({**spec, "n": 4})
    assert len(config_hash(spec)) == 16


def test_round_trip(tmp_path: Path) -> None:
    record = _record()
    write_record(record, tmp_path / "run")
    back = read_record(tmp_path / "run")
    assert back.spec == record.spec
    assert back.config_hash == record.config_hash
    assert back.metrics["infidelity"] == 3.1e-8
    assert back.metrics["gap"] == math.inf
    assert math.isnan(back.metrics["gamma"])
    np.testing.assert_array_equal(back.arrays["state"], record.arrays["state"])
    assert back.log == "built\n"
    assert back.provenance == record.provenance
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["schema"] == SCHEMA_VERSION


def test_failed_record_requires_error() -> None:
    with pytest.raises(ValueError, match="failed"):
        _record(status="failed")
    with pytest.raises(ValueError, match="failed"):
        _record(error="Traceback ...")
    assert _record(status="failed", error="Traceback ...").error


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        _record(status="done")


def test_tampered_spec_is_detected(tmp_path: Path) -> None:
    write_record(_record(), tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["spec"]["n"] = 6
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="config_hash"):
        read_record(tmp_path)


def test_provenance_records_the_environment() -> None:
    provenance = Provenance.capture()
    assert provenance.package_version
    assert set(provenance.dependencies) >= {"numpy", "scipy", "qiskit", "qiskit-aer"}
    assert provenance.dependencies["numpy"] == np.__version__
    assert provenance.git_commit is None or len(provenance.git_commit) == 40
