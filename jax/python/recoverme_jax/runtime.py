"""Recovery execution and benchmark orchestration around the JAX seed kernel."""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from recoverme_jax.crypto import JaxSeedDeriver
from recoverme_jax.domain import BenchmarkResult, DeviceInfo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

    import jax

    from recoverme_jax import _native


@dataclass(frozen=True, slots=True)
class RunResult:
    """Terminal recovery outcome returned to the CLI."""

    exhausted: bool
    interrupted: bool
    matches: int


def benchmark(
    session: _native.RecoverySession,
    mnemonic: np.ndarray[tuple[int], np.dtype[np.uint8]],
    info: DeviceInfo,
    device: jax.Device,
    batch_size: int,
) -> BenchmarkResult:
    """Measure compilation, seed derivation, CPU fingerprints, and complete checks."""
    sample = session.sample_batch(batch_size)
    deriver = JaxSeedDeriver.create(mnemonic.copy(), device, batch_size)
    candidates = np.asarray(sample.candidate_bytes, dtype=np.uint8)
    lengths = np.asarray(sample.lengths, dtype=np.uint16)

    compile_started = time.perf_counter()
    warmup = deriver.derive(candidates, lengths)
    compile_seconds = time.perf_counter() - compile_started
    warmup.fill(0)

    seed_started = time.perf_counter()
    seeds = deriver.derive(candidates, lengths)
    seed_seconds = time.perf_counter() - seed_started
    fingerprint_started = time.perf_counter()
    session.fingerprint_batch(seeds[: sample.count])
    fingerprint_seconds = time.perf_counter() - fingerprint_started
    seeds.fill(0)

    complete_seconds = seed_seconds + fingerprint_seconds
    candidates_checked = sample.count
    return BenchmarkResult(
        device=info,
        candidates=candidates_checked,
        batch_size=batch_size,
        compile_seconds=compile_seconds,
        seed_seconds=seed_seconds,
        fingerprint_seconds=fingerprint_seconds,
        seeds_per_second=candidates_checked / seed_seconds,
        fingerprints_per_second=candidates_checked / fingerprint_seconds,
        checks_per_second=candidates_checked / complete_seconds,
        measured_at_unix=int(time.time()),
    )


def run(
    session: _native.RecoverySession,
    deriver: JaxSeedDeriver,
    through: str,
) -> RunResult:
    """Run or resume batches, committing only after complete fingerprint verification."""
    stop = threading.Event()
    with _sigint_stop(stop):
        while not stop.is_set():
            batch = session.prepare_batch(through, deriver.batch_size)
            if batch is None:
                return RunResult(exhausted=True, interrupted=False, matches=0)
            candidates = np.asarray(batch.candidate_bytes, dtype=np.uint8)
            lengths = np.asarray(batch.lengths, dtype=np.uint16)
            seeds = deriver.derive(candidates, lengths)
            try:
                completion = session.complete_batch(batch.token, seeds)
            finally:
                seeds.fill(0)
            if completion.matches:
                return RunResult(
                    exhausted=False,
                    interrupted=False,
                    matches=completion.matches,
                )
        return RunResult(exhausted=False, interrupted=True, matches=0)


@contextmanager
def _sigint_stop(stop: threading.Event) -> Iterator[None]:
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(_signal: int, _frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
