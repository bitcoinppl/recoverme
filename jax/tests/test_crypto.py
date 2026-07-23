"""Trusted BIP39 seed and Coldcard fingerprint compatibility tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from mnemonic import Mnemonic
from recoverme_jax import _native
from recoverme_jax.crypto import MAX_PASSPHRASE_BYTES, JaxSeedDeriver
from recoverme_jax.devices import BackendUnavailableError, select_device
from recoverme_jax.domain import Backend

from tests.conftest import PUBLIC_TEST_MNEMONIC, PUBLIC_TEST_PASSPHRASE


def _derive(passphrases: list[str], backend: Backend) -> np.ndarray:
    _info, device = select_device(backend)
    mnemonic = np.frombuffer(PUBLIC_TEST_MNEMONIC.encode(), dtype=np.uint8).copy()
    deriver = JaxSeedDeriver.create(mnemonic, device, len(passphrases))
    candidates = np.zeros((len(passphrases), MAX_PASSPHRASE_BYTES), dtype=np.uint8)
    lengths = np.zeros(len(passphrases), dtype=np.uint16)
    for index, passphrase in enumerate(passphrases):
        encoded = passphrase.encode()
        candidates[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        lengths[index] = len(encoded)
    return deriver.derive(candidates, lengths)


def _trusted(passphrase: str) -> bytes:
    return Mnemonic("english").to_seed(PUBLIC_TEST_MNEMONIC, passphrase)


def test_cpu_seed_vectors_cover_case_length_padding_and_batch_boundaries() -> None:
    passphrases = [
        "",
        "lowercase",
        "TitleCase",
        "UPPERCASE",
        "mIxEdCaSe",
        "a" * 99,
        "b" * 100,
    ]
    actual_chunks = []
    for start in range(0, len(passphrases), 3):
        chunk = passphrases[start : start + 3]
        padded = [*chunk, *("" for _ in range(3 - len(chunk)))]
        actual_chunks.append(_derive(padded, Backend.CPU)[: len(chunk)])
    actual = np.concatenate(actual_chunks)

    assert [bytes(seed) for seed in actual] == [_trusted(value) for value in passphrases]


def test_public_fixture_fingerprints_match_coldcard_display_order(
    secret_files: tuple[Path, Path], tmp_path: Path
) -> None:
    mnemonic_file, words_file = secret_files
    empty = _native.RecoverySession.plan(
        str(mnemonic_file),
        str(words_file),
        "5436d724",
        str(tmp_path / "empty-state"),
        3,
        0,
        False,
    )
    full = _native.RecoverySession.plan(
        str(mnemonic_file),
        str(words_file),
        "997f3522",
        str(tmp_path / "full-state"),
        3,
        0,
        False,
    )
    seeds = np.stack(
        [
            np.frombuffer(_trusted(""), dtype=np.uint8),
            np.frombuffer(_trusted(PUBLIC_TEST_PASSPHRASE), dtype=np.uint8),
        ]
    )

    assert empty.fingerprint_batch(seeds) == [0]
    assert full.fingerprint_batch(seeds) == [1]


@pytest.mark.cuda
def test_cuda_seed_derivation_matches_trusted_bip39() -> None:
    try:
        actual = _derive(["", "TitleCase", "z" * 100], Backend.CUDA)
    except BackendUnavailableError as error:
        if os.environ.get("RECOVERME_REQUIRE_CUDA") == "1":
            pytest.fail(f"required CUDA validation unavailable: {error}")
        pytest.skip(f"CUDA validation unavailable: {error}")

    assert [bytes(seed) for seed in actual] == [
        _trusted(""),
        _trusted("TitleCase"),
        _trusted("z" * 100),
    ]
