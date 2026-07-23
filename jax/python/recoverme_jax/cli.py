"""Command-line interface for JAX-accelerated recoverme sessions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from recoverme_jax import _native
from recoverme_jax.benchmarks import BenchmarkStore, BenchmarkStoreError
from recoverme_jax.crypto import JaxSeedDeriver
from recoverme_jax.devices import BackendUnavailableError, available_devices, select_device
from recoverme_jax.domain import Backend, BenchmarkResult, DeviceInfo
from recoverme_jax.runtime import benchmark as benchmark_device
from recoverme_jax.runtime import run as run_recovery

if TYPE_CHECKING:
    import jax


def main() -> None:
    """Parse arguments, execute one command, and return a process exit status."""
    parser = _parser()
    arguments = parser.parse_args()
    try:
        arguments.handler(arguments)
    except (
        _native.RecoveryError,
        BackendUnavailableError,
        BenchmarkStoreError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recoverme-jax",
        description="JAX-accelerated deterministic Coldcard passphrase recovery",
    )
    commands = parser.add_subparsers(required=True)

    plan = commands.add_parser("plan", help="validate inputs and create shared state")
    _secret_files(plan)
    plan.add_argument("--fingerprint", required=True)
    plan.add_argument("--state-dir", type=Path, required=True)
    plan.add_argument("--neighbors", type=int, default=3)
    plan.add_argument("--max-replacements", type=int, default=2)
    plan.add_argument("--lowercase-already-tried", action="store_true")
    plan.add_argument("--order", choices=("written", "permuted"), default="permuted")
    plan.add_argument(
        "--spacing",
        choices=("concatenated", "between", "both", "coldcard"),
        default="concatenated",
    )
    plan.add_argument("--concatenated-already-tried", action="store_true")
    plan.set_defaults(handler=_plan)

    measure = commands.add_parser("benchmark", help="benchmark JAX recovery backends")
    _secret_files(measure)
    measure.add_argument("--state-dir", type=Path, required=True)
    _device_options(measure)
    measure.add_argument("--batch-size", type=int)
    measure.set_defaults(handler=_benchmark)

    run = commands.add_parser("run", help="run or resume through an explicit phase")
    _secret_files(run)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--through", required=True)
    _device_options(run)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--yes", action="store_true")
    run.set_defaults(handler=_run)

    reject = commands.add_parser("reject-match", help="reject a false four-byte collision")
    reject.add_argument("--state-dir", type=Path, required=True)
    reject.add_argument("--match-id", required=True)
    reject.set_defaults(handler=_reject)
    return parser


def _secret_files(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mnemonic-file", type=Path, required=True)
    recipes = parser.add_mutually_exclusive_group(required=True)
    recipes.add_argument("--words-file", type=Path)
    recipes.add_argument("--recipe-file", type=Path)


def _device_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=tuple(Backend), default=Backend.AUTO)
    parser.add_argument("--device-index", type=int, default=0)


def _plan(arguments: argparse.Namespace) -> None:
    planner = (
        _native.RecoverySession.plan_recipe
        if arguments.recipe_file is not None
        else _native.RecoverySession.plan
    )
    input_file = arguments.recipe_file or arguments.words_file
    session = planner(
        str(arguments.mnemonic_file),
        str(input_file),
        arguments.fingerprint,
        str(arguments.state_dir),
        arguments.neighbors,
        arguments.max_replacements,
        arguments.lowercase_already_tried,
        arguments.order,
        arguments.spacing,
        arguments.concatenated_already_tried,
    )
    print("Nearest BIP39 words:")
    for written, raw_neighbors in session.neighbor_suggestions():
        neighbors = ", ".join(f"{word} [d={distance}]" for word, distance in raw_neighbors)
        print(f"  {written}: {neighbors}")
    print("Search phases:")
    for phase, count in session.phase_summaries():
        probability = -math.expm1(-(count / 4_294_967_296.0)) * 100
        print(f"  {phase:<20} {_format_count(count):>20}  XFP collision {probability:>8.4f}%")
    info, _device = select_device(Backend.AUTO)
    _print_device(info)
    print(f"State: {arguments.state_dir}")


def _benchmark(arguments: argparse.Namespace) -> None:
    session = _open_session(arguments)
    mnemonic = np.asarray(session.take_mnemonic_bytes(), dtype=np.uint8)
    store = BenchmarkStore(arguments.state_dir)
    try:
        selections = _selections(Backend(arguments.backend), arguments.device_index)
        for info, device in selections:
            batch_size = arguments.batch_size or _default_batch_size(info.backend)
            result = benchmark_device(session, mnemonic, info, device, batch_size)
            store.record(result)
            _print_benchmark(result)
    finally:
        mnemonic.fill(0)


def _run(arguments: argparse.Namespace) -> None:
    session = _open_session(arguments)
    if session.has_pending_matches():
        raise _native.RecoveryError(
            "pending XFP matches must be verified or rejected before resuming"
        )
    mnemonic = np.asarray(session.take_mnemonic_bytes(), dtype=np.uint8)
    store = BenchmarkStore(arguments.state_dir)
    try:
        info, device, batch_size = _select_for_run(
            session,
            mnemonic,
            store,
            Backend(arguments.backend),
            arguments.device_index,
            arguments.batch_size,
        )
        _print_device(info)
        total = _count_through(session.phase_summaries(), arguments.through)
        remaining = max(0, total - session.completed())
        print(f"Authorized through: {arguments.through}")
        print(f"Remaining candidates: {_format_count(remaining)}")
        print(f"Expected random XFP hits: {total / 4_294_967_296.0:.4f}")
        if not arguments.yes and not _confirm("Start or resume this search? [y/N] "):
            print("Cancelled")
            return

        deriver = JaxSeedDeriver.create(mnemonic.copy(), device, batch_size)
        result = run_recovery(session, deriver, arguments.through)
        if result.matches:
            print(f"Found {result.matches} XFP candidate(s); verify manually on the Coldcard")
            for match_id, _phase, passphrase, words in session.pending_matches():
                print(f"Match ID: {match_id}")
                print(f"Exact passphrase: {passphrase}")
                print(f"Readable words: {' '.join(words)}")
            print(
                "A four-byte XFP can collide; do not trust this result "
                "without Coldcard verification"
            )
        elif result.interrupted:
            print("Stopped cleanly after the last completed batch")
        else:
            print("Authorized phases exhausted without a pending XFP match")
    finally:
        mnemonic.fill(0)


def _reject(arguments: argparse.Namespace) -> None:
    _native.reject_match(str(arguments.state_dir), arguments.match_id)
    print(f"Rejected XFP collision {arguments.match_id}; the next run will resume")


def _open_session(arguments: argparse.Namespace) -> _native.RecoverySession:
    opener = (
        _native.RecoverySession.open_recipe
        if arguments.recipe_file is not None
        else _native.RecoverySession.open
    )
    input_file = arguments.recipe_file or arguments.words_file
    return opener(str(arguments.mnemonic_file), str(input_file), str(arguments.state_dir))


def _select_for_run(
    session: _native.RecoverySession,
    mnemonic: np.ndarray[tuple[int], np.dtype[np.uint8]],
    store: BenchmarkStore,
    backend: Backend,
    device_index: int,
    requested_batch_size: int | None,
) -> tuple[DeviceInfo, jax.Device, int]:
    selections = _selections(backend, device_index)
    measured: list[tuple[BenchmarkResult, jax.Device]] = []
    for info, device in selections:
        batch_size = requested_batch_size or _default_batch_size(info.backend)
        record = store.latest(info, batch_size)
        if record is None:
            record = benchmark_device(session, mnemonic, info, device, batch_size)
            store.record(record)
            _print_benchmark(record)
        measured.append((record, device))
    record, device = max(measured, key=lambda entry: entry[0].checks_per_second)
    return record.device, device, record.batch_size


def _selections(backend: Backend, device_index: int) -> list[tuple[DeviceInfo, jax.Device]]:
    if backend is not Backend.AUTO:
        return [select_device(backend, device_index)]
    selections = list(available_devices())
    if not selections:
        raise BackendUnavailableError("JAX detected no supported CPU or NVIDIA device")
    return selections


def _default_batch_size(backend: Backend) -> int:
    return 4_096 if backend is Backend.CUDA else 64


def _print_device(info: DeviceInfo) -> None:
    print(
        "JAX device: "
        f"backend={info.backend.value}, platform={info.platform}, "
        f"kind={info.device_kind}, id={info.device_id}, jax={info.jax_version}"
    )


def _print_benchmark(result: BenchmarkResult) -> None:
    _print_device(result.device)
    print(
        f"  {result.seeds_per_second:.1f} seeds/s, "
        f"{result.fingerprints_per_second:.1f} fingerprints/s, "
        f"{result.checks_per_second:.1f} complete checks/s, "
        f"batch={result.batch_size}, compile+warmup={result.compile_seconds:.3f}s"
    )


def _count_through(summaries: list[tuple[str, int]], through: str) -> int:
    total = 0
    for phase, count in summaries:
        total += count
        if phase == through:
            return total
    raise ValueError(f"phase is not enabled in this recovery plan: {through}")


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in {"y", "yes"}


def _format_count(value: int) -> str:
    return f"{value:,}"


if __name__ == "__main__":
    main()
