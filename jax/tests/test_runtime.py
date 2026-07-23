"""Recovery loop interruption and benchmark sidecar tests."""

from __future__ import annotations

import os
import signal
import stat
from pathlib import Path

import numpy as np
import pytest
from recoverme_jax import _native
from recoverme_jax.benchmarks import BenchmarkStore, BenchmarkStoreError
from recoverme_jax.domain import Backend, BenchmarkResult, DeviceInfo
from recoverme_jax.runtime import run


class InterruptingDeriver:
    """Fixed-size test deriver that requests shutdown during its first batch."""

    batch_size = 1

    def derive(self, _candidates: np.ndarray, _lengths: np.ndarray) -> np.ndarray:
        """Request SIGINT and return one structurally valid seed."""
        os.kill(os.getpid(), signal.SIGINT)
        return np.zeros((1, 64), dtype=np.uint8)


def test_sigint_finishes_and_commits_the_current_batch(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, words_file = secret_files
    state_dir = tmp_path / "state"
    session = _native.RecoverySession.plan(
        str(mnemonic_file),
        str(words_file),
        "ffffffff",
        str(state_dir),
        3,
        0,
        False,
    )

    result = run(session, InterruptingDeriver(), "written-lower")  # type: ignore[arg-type]

    assert result.interrupted
    assert session.completed() == 1
    reopened = _native.RecoverySession.open(str(mnemonic_file), str(words_file), str(state_dir))
    assert reopened.completed() == 1


def _benchmark() -> BenchmarkResult:
    info = DeviceInfo(Backend.CPU, "cpu", "test cpu", 0, "test")
    return BenchmarkResult(info, 8, 8, 1.0, 2.0, 3.0, 4.0, 5.0, 1.6, 123)


@pytest.mark.skipif(os.name != "posix", reason="owner-only modes require POSIX")
def test_benchmark_sidecar_is_atomic_owner_only_and_versioned(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store = BenchmarkStore(state_dir)
    result = _benchmark()
    store.record(result)
    path = state_dir / "jax-benchmarks-v1.json"

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert BenchmarkStore(state_dir).latest(result.device, 8) == result
    assert not list(state_dir.glob(".jax-benchmarks-v1.json.tmp-*"))

    path.chmod(0o644)
    with pytest.raises(BenchmarkStoreError, match="owner-only"):
        BenchmarkStore(state_dir)
