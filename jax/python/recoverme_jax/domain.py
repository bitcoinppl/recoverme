"""Typed public domain models for the JAX recovery frontend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Backend(StrEnum):
    """User-selectable JAX execution backend."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Stable description of one selected JAX device."""

    backend: Backend
    platform: str
    device_kind: str
    device_id: int
    jax_version: str

    @property
    def key(self) -> str:
        """Return an environment key suitable for benchmark selection."""
        return (
            f"{self.backend.value}:{self.platform}:{self.device_kind}:"
            f"{self.device_id}:jax-{self.jax_version}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measured seed, fingerprint, and complete-check throughput."""

    device: DeviceInfo
    candidates: int
    batch_size: int
    compile_seconds: float
    seed_seconds: float
    fingerprint_seconds: float
    seeds_per_second: float
    fingerprints_per_second: float
    checks_per_second: float
    measured_at_unix: int


@dataclass(frozen=True, slots=True)
class PhaseSummary:
    """Exact unique candidate count for one recovery phase."""

    phase: str
    count: int


@dataclass(frozen=True, slots=True)
class NeighborWord:
    """One BIP39 neighbor and its edit distance."""

    word: str
    distance: int


@dataclass(frozen=True, slots=True)
class NeighborSuggestion:
    """Ranked BIP39 neighbors for a written token."""

    written: str
    neighbors: tuple[NeighborWord, ...]
