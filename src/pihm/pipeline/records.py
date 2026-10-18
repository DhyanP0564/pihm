r"""Run records, schema version 1: strict JSON, NPZ arrays and provenance.

Every pipeline run writes one directory holding

``meta.json``
    The run specification, its configuration hash, the status, provenance, timings and, for a
    failed run, the traceback.
``metrics.json``
    Scalar results, in strict JSON.
``arrays.npz``
    Array results (optional).
``log.txt``
    The run's log (optional).

Strict JSON means standard JSON (RFC 8259) only. Python's :mod:`json` writes the non-standard
tokens ``NaN``, ``Infinity`` and ``-Infinity`` by default, which several readers reject and
which once corrupted a results table. Here non-finite floats
are written as tagged objects instead, for example ``{"$float": "nan"}``, and are restored on
reading, so a record round-trips losslessly.

Examples
--------
>>> text = dumps_strict({"infidelity": float("nan"), "degree": 912})
>>> text
'{"degree": 912, "infidelity": {"$float": "nan"}}'
>>> loads_strict(text)["degree"]
912
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, get_args

import numpy as np
from numpy.typing import NDArray

import pihm

__all__ = [
    "RUN_MODULE",
    "SCHEMA_VERSION",
    "Provenance",
    "Record",
    "Status",
    "config_hash",
    "dumps_strict",
    "loads_strict",
    "read_record",
    "run_modules",
    "source_digest",
    "write_record",
]

#: Version of the on-disk record layout; bumped on any incompatible change.
SCHEMA_VERSION: Final[int] = 1

#: Outcome of a run. ``infeasible`` means the run was predicted to exceed its resource class and
#: was not attempted; ``skipped`` means an ``ok`` record with the same configuration hash exists.
Status: TypeAlias = Literal["ok", "failed", "infeasible", "skipped"]

#: A value that :func:`dumps_strict` accepts.
JSONValue: TypeAlias = Any

#: Packages whose versions are recorded in every record's provenance.
_TRACKED_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "numpy",
    "scipy",
    "qiskit",
    "qiskit-aer",
    "pyyaml",
)

#: Slurm environment variables recorded when present.
_SLURM_VARIABLES: Final[tuple[str, ...]] = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_PARTITION",
)

_FLOAT_TAG: Final[str] = "$float"
_NONFINITE_NAMES: Final[dict[str, float]] = {
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}


def _encode(value: JSONValue) -> JSONValue:
    """Convert ``value`` to plain JSON types, tagging non-finite floats."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return number
        name = "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")
        return {_FLOAT_TAG: name}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _encode(value.item())
    if isinstance(value, Mapping):
        encoded: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be str, got {type(key).__name__}")
            if key == _FLOAT_TAG:
                raise ValueError(f"the key {_FLOAT_TAG!r} is reserved for non-finite floats")
            encoded[key] = _encode(item)
        return encoded
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    raise TypeError(
        f"cannot write {type(value).__name__} to strict JSON; put arrays and complex values "
        "in arrays.npz instead"
    )


def _decode_hook(obj: dict[str, JSONValue]) -> JSONValue:
    """Restore a tagged non-finite float written by :func:`_encode`."""
    if set(obj) == {_FLOAT_TAG}:
        name = obj[_FLOAT_TAG]
        if name not in _NONFINITE_NAMES:
            raise ValueError(f"unknown tagged float {name!r}")
        return _NONFINITE_NAMES[name]
    return obj


def _reject_constant(token: str) -> None:
    """Refuse the non-standard constants ``NaN``, ``Infinity`` and ``-Infinity``."""
    raise ValueError(f"non-standard JSON constant {token!r}; records must be strict JSON")


def dumps_strict(obj: JSONValue, *, indent: int | None = None) -> str:
    """Serialise ``obj`` as strict JSON, with keys sorted.

    Parameters
    ----------
    obj
        Nested mappings, lists and tuples of ``None``, ``bool``, ``int``, ``float`` and ``str``.
        NumPy scalars and zero-dimensional arrays are converted to their Python equivalents.
    indent
        Indentation passed to :func:`json.dumps`; ``None`` gives the compact form.

    Returns
    -------
    str
        Standard JSON text. Non-finite floats appear as ``{"$float": "nan" | "inf" | "-inf"}``.

    Raises
    ------
    TypeError
        If ``obj`` contains a type that has no strict JSON form (for example an array or a
        complex number) or a mapping with a non-string key.
    ValueError
        If a mapping uses the reserved key ``"$float"``.
    """
    return json.dumps(_encode(obj), allow_nan=False, sort_keys=True, indent=indent)


def loads_strict(text: str) -> JSONValue:
    """Parse strict JSON written by :func:`dumps_strict`, restoring non-finite floats.

    Parameters
    ----------
    text
        JSON text.

    Returns
    -------
    object
        The decoded value.

    Raises
    ------
    ValueError
        If ``text`` contains a bare ``NaN``, ``Infinity`` or ``-Infinity`` token.
    """
    return json.loads(text, object_hook=_decode_hook, parse_constant=_reject_constant)


def config_hash(spec: Mapping[str, JSONValue]) -> str:
    """Return the configuration hash of a run specification.

    The hash is the first 16 hexadecimal digits of the SHA-256 of the canonical strict JSON
    form (sorted keys, no whitespace). It is therefore independent of key order and changes
    whenever any value changes.

    Parameters
    ----------
    spec
        The run specification, as a JSON-serialisable mapping.

    Returns
    -------
    str
        Sixteen lower-case hexadecimal digits.

    Examples
    --------
    >>> config_hash({"n": 3, "regime": "descriptor"}) == config_hash(
    ...     {"regime": "descriptor", "n": 3}
    ... )
    True
    """
    canonical = json.dumps(_encode(spec), allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command and return its stripped output, or ``None`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


#: The module whose imports, followed through the package, are the source a run executes
#: (:func:`run_modules`).
RUN_MODULE: Final[str] = "pihm.pipeline.stages"


def _module_file(package: Path, name: str) -> Path | None:
    """Return the file of the package's module ``name``, or ``None`` if it is not a module.

    Every part of the path must match in case, as Python's imports require on any filesystem: on
    one that ignores case (macOS's), ``assemble/Residual.py`` would otherwise name the class
    ``Residual`` imported from ``assemble/residual.py`` as a module of its own.
    """
    parts = name.split(".")[1:]
    candidates = [(*parts, "__init__.py")]
    if parts:
        candidates.append((*parts[:-1], f"{parts[-1]}.py"))
    for relative in candidates:
        if _exact_file(package, relative):
            return package.joinpath(*relative)
    return None


def _exact_file(root: Path, parts: tuple[str, ...]) -> bool:
    """Return whether ``root`` holds the file ``parts``, each part named in its exact case."""
    for part in parts:
        if not root.is_dir() or part not in os.listdir(root):
            return False
        root = root / part
    return root.is_file()


def run_modules(package: Path | None = None) -> dict[str, Path]:
    """Return the source a run executes: :data:`RUN_MODULE` and every package module it imports.

    The imports are read from the source, not from the running process, and followed
    transitively, each module's parent packages included, since importing a module runs them.
    What a run cannot import, the analysis (the summaries, the catalogue, the audit, the readers
    of the reports, the plots), is left out: it can change without touching a
    record.
    """
    root = package if package is not None else Path(pihm.__file__).resolve().parent
    found: dict[str, Path] = {}
    pending = [RUN_MODULE]
    while pending:
        name = pending.pop()
        inside = name.split(".")[0] == "pihm" and name not in found
        path = _module_file(root, name) if inside else None
        if path is None:
            continue
        found[name] = path
        parts = name.split(".")
        pending += [".".join(parts[:i]) for i in range(1, len(parts))]
        here = parts if path.name == "__init__.py" else parts[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = here[: len(here) - node.level + 1] if node.level else []
                module = ".".join(base + ([node.module] if node.module else []))
                pending += [module, *(f"{module}.{alias.name}" for alias in node.names)]
    return found


def source_digest(package: Path | None = None) -> str:
    """Return sixteen hexadecimal digits of the SHA-256 of the source a run executes.

    Each module of :func:`run_modules` enters as its path within the package and its bytes, in
    the order of its name, so the digest is the same in a checkout and in a synced copy, and
    changes with any change a record could depend on.
    """
    root = package if package is not None else Path(pihm.__file__).resolve().parent
    digest = hashlib.sha256()
    for _name, path in sorted(run_modules(root).items()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def _distribution_version(name: str) -> str:
    """Return the installed version of distribution ``name``, or ``"absent"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


@dataclass(frozen=True)
class Provenance:
    """Where, when and with what code a record was produced.

    Attributes
    ----------
    git_commit
        Commit hash of the working tree, or ``None`` outside a git checkout.
    git_dirty
        Whether the working tree had uncommitted changes (``None`` if unknown).
    package_version
        The installed ``pihm.__version__``.
    python_version
        Interpreter version, e.g. ``"3.11.8"``.
    platform
        Operating system and machine, from :func:`platform.platform`.
    hostname
        Network name of the machine.
    dependencies
        Versions of the tracked third-party distributions.
    slurm
        The Slurm job variables present in the environment (empty off the cluster).
    source_digest
        :func:`source_digest` of the source that ran; ``None`` where unknown.
    created_utc
        ISO 8601 timestamp in UTC.
    """

    git_commit: str | None
    git_dirty: bool | None
    package_version: str
    python_version: str
    platform: str
    hostname: str
    dependencies: Mapping[str, str]
    slurm: Mapping[str, str]
    created_utc: str
    source_digest: str | None = None

    @classmethod
    def capture(cls, repository: Path | None = None) -> Provenance:
        """Capture the provenance of the current process.

        Parameters
        ----------
        repository
            Directory inside the git checkout to describe; defaults to this package's source.

        Returns
        -------
        Provenance
            A snapshot of the environment.
        """
        where = repository if repository is not None else Path(__file__).resolve().parent
        commit = _git(["rev-parse", "HEAD"], where)
        status = _git(["status", "--porcelain", "--untracked-files=no"], where)
        return cls(
            git_commit=commit,
            git_dirty=None if status is None else bool(status),
            package_version=pihm.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            hostname=platform.node(),
            dependencies={name: _distribution_version(name) for name in _TRACKED_DISTRIBUTIONS},
            slurm={key: os.environ[key] for key in _SLURM_VARIABLES if key in os.environ},
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            source_digest=source_digest(),
        )

    def to_json(self) -> dict[str, JSONValue]:
        """Return the provenance as a JSON-serialisable mapping."""
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "dependencies": dict(self.dependencies),
            "slurm": dict(self.slurm),
            "created_utc": self.created_utc,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, JSONValue]) -> Provenance:
        """Rebuild a :class:`Provenance` from :meth:`to_json` output."""
        return cls(
            git_commit=data["git_commit"],
            git_dirty=data["git_dirty"],
            package_version=data["package_version"],
            python_version=data["python_version"],
            platform=data["platform"],
            hostname=data["hostname"],
            dependencies=dict(data["dependencies"]),
            slurm=dict(data["slurm"]),
            created_utc=data["created_utc"],
            source_digest=data.get("source_digest"),
        )


@dataclass(frozen=True)
class Record:
    """The complete result of one run, as written to and read from disk.

    Attributes
    ----------
    spec
        The run specification (JSON-serialisable); :attr:`config_hash` is derived from it.
    status
        The run's outcome.
    metrics
        Scalar results (strict JSON).
    provenance
        Where and with what code the run was produced.
    timings
        Wall-clock seconds per named step.
    peak_memory_gib
        The process's peak resident memory when the run ended, in GiB: the run's own where it has
        a process to itself (a Slurm array task), the largest of the runs so far where several
        share one. ``None`` where unknown.
    arrays
        Array results, stored in ``arrays.npz``.
    log
        The run's log text, stored in ``log.txt``.
    error
        The traceback of a failed run, otherwise ``None``.
    """

    spec: Mapping[str, JSONValue]
    status: Status
    metrics: Mapping[str, JSONValue]
    provenance: Provenance
    timings: Mapping[str, float] = field(default_factory=dict)
    peak_memory_gib: float | None = None
    arrays: Mapping[str, NDArray[Any]] = field(default_factory=dict)
    log: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate the status and the consistency of ``error`` with it."""
        if self.status not in get_args(Status):
            raise ValueError(f"status must be one of {get_args(Status)}, got {self.status!r}")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("a failed record needs an error, and only a failed record has one")

    @property
    def config_hash(self) -> str:
        """The configuration hash of :attr:`spec` (see :func:`config_hash`)."""
        return config_hash(self.spec)


def write_record(record: Record, directory: Path) -> Path:
    """Write ``record`` to ``directory``, creating it if needed.

    Parameters
    ----------
    record
        The record to write.
    directory
        Destination directory; existing record files in it are overwritten.

    Returns
    -------
    pathlib.Path
        ``directory``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA_VERSION,
        "spec": record.spec,
        "config_hash": record.config_hash,
        "status": record.status,
        "provenance": record.provenance.to_json(),
        "timings": record.timings,
        "peak_memory_gib": record.peak_memory_gib,
        "error": record.error,
    }
    (directory / "meta.json").write_text(dumps_strict(meta, indent=2) + "\n")
    (directory / "metrics.json").write_text(dumps_strict(record.metrics, indent=2) + "\n")
    if record.arrays:
        np.savez_compressed(directory / "arrays.npz", allow_pickle=False, **record.arrays)
    if record.log:
        (directory / "log.txt").write_text(record.log)
    return directory


def read_record(directory: Path) -> Record:
    """Read a record written by :func:`write_record`.

    Parameters
    ----------
    directory
        The record directory.

    Returns
    -------
    Record
        The record, with non-finite metrics restored.

    Raises
    ------
    ValueError
        If the schema version is not :data:`SCHEMA_VERSION`, or if the stored configuration hash
        does not match the stored specification.
    """
    meta = loads_strict((directory / "meta.json").read_text())
    if meta.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"record schema {meta.get('schema')!r} is not {SCHEMA_VERSION}")
    if config_hash(meta["spec"]) != meta["config_hash"]:
        raise ValueError("stored config_hash does not match the stored spec")
    arrays: dict[str, NDArray[Any]] = {}
    npz = directory / "arrays.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    log_path = directory / "log.txt"
    return Record(
        spec=meta["spec"],
        status=meta["status"],
        metrics=loads_strict((directory / "metrics.json").read_text()),
        provenance=Provenance.from_json(meta["provenance"]),
        timings=meta["timings"],
        peak_memory_gib=meta.get("peak_memory_gib"),
        arrays=arrays,
        log=log_path.read_text() if log_path.exists() else "",
        error=meta["error"],
    )
