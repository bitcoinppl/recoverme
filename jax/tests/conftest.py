"""Shared public BIP39 recovery fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

PUBLIC_TEST_MNEMONIC = (
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon "
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon "
    "abandon abandon abandon art"
)
PUBLIC_TEST_PASSPHRASE = "alphabriskcactusdaringeagerfabricgadget"
PUBLIC_TEST_WORDS = ("alpha", "brisk", "cactus", "daring", "eager", "fabric", "gadget")


@pytest.fixture
def secret_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create owner-only files containing only public test vectors."""
    mnemonic_file = tmp_path / "mnemonic.txt"
    words_file = tmp_path / "words.txt"
    mnemonic_file.write_text(f"{PUBLIC_TEST_MNEMONIC}\n")
    words_file.write_text("alpha\nbrisk\n")
    mnemonic_file.chmod(0o600)
    words_file.chmod(0o600)
    return mnemonic_file, words_file


def write_secret_files(directory: Path, words: tuple[str, ...]) -> tuple[Path, Path]:
    """Write an owner-only public mnemonic and chosen test words."""
    mnemonic_file = directory / "mnemonic.txt"
    words_file = directory / "words.txt"
    mnemonic_file.write_text(f"{PUBLIC_TEST_MNEMONIC}\n")
    words_file.write_text("".join(f"{word}\n" for word in words))
    mnemonic_file.chmod(0o600)
    words_file.chmod(0o600)
    return mnemonic_file, words_file
