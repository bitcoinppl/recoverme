"""Rust candidate planner and shared-checkpoint integration tests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from mnemonic import Mnemonic
from recoverme_jax import _native

from tests.conftest import (
    PUBLIC_TEST_MNEMONIC,
    PUBLIC_TEST_PASSPHRASE,
    PUBLIC_TEST_WORDS,
    write_secret_files,
)


def _plan(
    mnemonic_file: Path,
    words_file: Path,
    state_dir: Path,
    *,
    fingerprint: str = "ffffffff",
    lowercase_already_tried: bool = False,
) -> _native.RecoverySession:
    return _native.RecoverySession.plan(
        str(mnemonic_file),
        str(words_file),
        fingerprint,
        str(state_dir),
        3,
        0,
        lowercase_already_tried,
    )


def _candidate_id(passphrase: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"recoverme-candidate\0")
    digest.update((3).to_bytes(4, "little"))
    digest.update(passphrase.encode())
    return digest.hexdigest()


def test_candidate_order_ids_counts_and_cursor_resume_match_rust_core(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, words_file = secret_files
    session = _plan(mnemonic_file, words_file, tmp_path / "state")

    first, cursor = session.enumerate_candidates("written-case", 3)
    second, _cursor = session.enumerate_candidates("written-case", 5, cursor)
    together, _cursor = session.enumerate_candidates("written-case", 8)

    assert [record[2] for record in together[:4]] == [
        "alphabrisk",
        "briskalpha",
        "AlphaBrisk",
        "ALPHABRISK",
    ]
    assert first + second == together
    assert len({record[2].encode() for record in together}) == len(together)
    assert all(" " not in record[2] for record in together)
    assert all(record[0] == _candidate_id(record[2]) for record in together)
    assert session.phase_summaries()[:2] == [
        ("written-lower", 2),
        ("written-case", 16),
    ]


def test_lowercase_already_tried_removes_every_lowercase_phase(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, words_file = secret_files
    session = _plan(
        mnemonic_file,
        words_file,
        tmp_path / "state",
        lowercase_already_tried=True,
    )
    records, _cursor = session.enumerate_candidates("written-case", 2)

    assert session.phase_summaries() == [("written-case", 16)]
    assert [record[2] for record in records] == ["AlphaBrisk", "ALPHABRISK"]


def test_recipe_and_coldcard_spacing_match_the_rust_candidate_model(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, _words_file = secret_files
    recipe_file = tmp_path / "recipe.toml"
    recipe_file.write_text(
        'version = 1\n\n[[slots]]\nalternatives = ["alpha", "alps"]\n\n'
        '[[slots]]\nalternatives = ["brisk"]\noptional = true\n'
    )
    recipe_file.chmod(0o600)
    session = _native.RecoverySession.plan_recipe(
        str(mnemonic_file),
        str(recipe_file),
        "ffffffff",
        str(tmp_path / "recipe-state"),
        3,
        0,
        False,
        "written",
        "coldcard",
        False,
    )

    records, _cursor = session.enumerate_candidates("written-lower", 100)
    passphrases = {record[2] for record in records}

    assert session.settings()[5:] == ("written", "coldcard", False)
    assert passphrases == {
        "alphabrisk",
        "alpha brisk",
        " alphabrisk",
        " alpha brisk",
        "alpsbrisk",
        "alps brisk",
        " alpsbrisk",
        " alps brisk",
        "alpha",
        " alpha",
        "alps",
        " alps",
    }


def test_prepared_batch_is_not_checkpointed_until_fingerprints_complete(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, words_file = secret_files
    state_dir = tmp_path / "state"
    session = _plan(mnemonic_file, words_file, state_dir)
    runtime_path = state_dir / "runtime.json"
    before = runtime_path.read_bytes()

    batch = session.prepare_batch("written-lower", 1)
    assert batch is not None
    assert runtime_path.read_bytes() == before

    reopened = _native.RecoverySession.open(str(mnemonic_file), str(words_file), str(state_dir))
    replay = reopened.prepare_batch("written-lower", 1)
    assert replay is not None
    assert np.array_equal(batch.candidate_bytes, replay.candidate_bytes)

    completion = reopened.complete_batch(replay.token, np.zeros((1, 64), dtype=np.uint8))
    assert completion.checked == 1
    assert completion.completed == 1
    assert json.loads(runtime_path.read_bytes())["cursor"]["completed"] == "1"


def test_match_rejection_resumes_after_exact_verified_candidate(tmp_path: Path) -> None:
    mnemonic_file, words_file = write_secret_files(tmp_path, PUBLIC_TEST_WORDS)
    state_dir = tmp_path / "state"
    session = _plan(
        mnemonic_file,
        words_file,
        state_dir,
        fingerprint="997f3522",
    )
    batch = session.prepare_batch("written-lower", 1)
    assert batch is not None
    assert bytes(batch.candidate_bytes[0, : batch.lengths[0]]) == PUBLIC_TEST_PASSPHRASE.encode()
    seed = np.frombuffer(
        Mnemonic("english").to_seed(PUBLIC_TEST_MNEMONIC, PUBLIC_TEST_PASSPHRASE),
        dtype=np.uint8,
    ).reshape(1, 64)

    completion = session.complete_batch(batch.token, seed)
    assert completion.matches == 1
    match_id, _phase, passphrase, words = session.pending_matches()[0]
    assert passphrase == PUBLIC_TEST_PASSPHRASE
    assert words == list(PUBLIC_TEST_WORDS)

    _native.reject_match(str(state_dir), match_id)
    resumed = _native.RecoverySession.open(str(mnemonic_file), str(words_file), str(state_dir))
    assert not resumed.has_pending_matches()
    next_batch = resumed.prepare_batch("written-lower", 1)
    assert next_batch is not None
    next_value = bytes(next_batch.candidate_bytes[0, : next_batch.lengths[0]])
    assert next_value != PUBLIC_TEST_PASSPHRASE.encode()
    assert resumed.completed() == 1


@pytest.mark.skipif(os.name != "posix", reason="owner-only modes require POSIX")
def test_secret_and_state_permissions_are_enforced(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, words_file = secret_files
    mnemonic_file.chmod(0o644)
    with pytest.raises(_native.RecoveryError, match="owner-only"):
        _plan(mnemonic_file, words_file, tmp_path / "rejected-state")

    mnemonic_file.chmod(0o600)
    state_dir = tmp_path / "state"
    _plan(mnemonic_file, words_file, state_dir)
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((state_dir / "manifest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((state_dir / "runtime.json").stat().st_mode) == 0o600


def test_invalid_secret_values_are_redacted(tmp_path: Path) -> None:
    mnemonic_file = tmp_path / "mnemonic.txt"
    words_file = tmp_path / "words.txt"
    secret_marker = "never-print-this-invalid-secret"
    mnemonic_file.write_text(f"{secret_marker}\n")
    words_file.write_text("alpha\n")
    mnemonic_file.chmod(0o600)
    words_file.chmod(0o600)

    with pytest.raises(_native.RecoveryError) as raised:
        _plan(mnemonic_file, words_file, tmp_path / "state")
    assert secret_marker not in str(raised.value)
