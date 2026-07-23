"""Versioned owner-only benchmark sidecar for JAX-specific measurements."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from pathlib import Path

from recoverme_jax.domain import Backend, BenchmarkResult, DeviceInfo

_SCHEMA_VERSION: Final = 1
_FILE_NAME: Final = "jax-benchmarks-v1.json"


class BenchmarkStoreError(RuntimeError):
    """Raised when the benchmark sidecar is invalid or insecure."""


class BenchmarkStore:
    """Typed benchmark history independent from Rust backend records."""

    def __init__(self, state_dir: Path) -> None:
        """Load benchmark records rooted in one recovery state directory."""
        self._state_dir = state_dir
        self._path = state_dir / _FILE_NAME
        self._records = self._load()

    def latest(self, device: DeviceInfo, batch_size: int | None = None) -> BenchmarkResult | None:
        """Return the newest compatible record for a device and optional batch size."""
        return next(
            (
                record
                for record in reversed(self._records)
                if record.device.key == device.key
                and (batch_size is None or record.batch_size == batch_size)
            ),
            None,
        )

    def record(self, result: BenchmarkResult) -> None:
        """Append and atomically persist one completed measurement."""
        self._records.append(result)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "records": [_serialize(record) for record in self._records],
        }
        _atomic_write(self._path, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")

    def _load(self) -> list[BenchmarkResult]:
        if not self._path.exists():
            return []
        _require_owner_only(self._path)
        try:
            payload = cast("dict[str, Any]", json.loads(self._path.read_bytes()))
            if payload.get("schema_version") != _SCHEMA_VERSION:
                raise BenchmarkStoreError("unsupported JAX benchmark schema")
            raw_records = cast("list[dict[str, Any]]", payload["records"])
            return [_deserialize(record) for record in raw_records]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise BenchmarkStoreError("invalid JAX benchmark sidecar") from error


def _serialize(result: BenchmarkResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["device"]["backend"] = result.device.backend.value
    return payload


def _deserialize(payload: dict[str, Any]) -> BenchmarkResult:
    device_payload = cast("dict[str, Any]", payload["device"])
    device = DeviceInfo(
        backend=Backend(str(device_payload["backend"])),
        platform=str(device_payload["platform"]),
        device_kind=str(device_payload["device_kind"]),
        device_id=int(device_payload["device_id"]),
        jax_version=str(device_payload["jax_version"]),
    )
    return BenchmarkResult(
        device=device,
        candidates=int(payload["candidates"]),
        batch_size=int(payload["batch_size"]),
        compile_seconds=float(payload["compile_seconds"]),
        seed_seconds=float(payload["seed_seconds"]),
        fingerprint_seconds=float(payload["fingerprint_seconds"]),
        seeds_per_second=float(payload["seeds_per_second"]),
        fingerprints_per_second=float(payload["fingerprints_per_second"]),
        checks_per_second=float(payload["checks_per_second"]),
        measured_at_unix=int(payload["measured_at_unix"]),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name == "posix":
            path.chmod(0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_owner_only(path: Path) -> None:
    if os.name != "posix":
        raise BenchmarkStoreError("owner-only state validation requires a POSIX platform")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise BenchmarkStoreError(f"protected file must be owner-only: {path}")
