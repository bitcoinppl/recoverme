# recoverme-jax

`recoverme-jax` runs BIP39 PBKDF2-HMAC-SHA512 in fixed-size JIT-compiled JAX batches while reusing recoverme's Rust candidate planner, checkpoint format, and BIP32 fingerprint implementation.

Every passphrase supplied to BIP39 is the exact concatenation of its words with **no spaces**. Spaces appear only in the readable match display.

## Security

- Work only on an offline computer you control.
- Put the mnemonic and written passphrase words in separate owner-only files, with one written word per line:

  ```sh
  chmod 600 mnemonic.txt written-words.txt
  ```

- The mnemonic is never accepted as a command-line argument or environment variable.
- The shared state directory is mode `0700`; state files are mode `0600`.
- A pending match record contains the exact passphrase. Protect the state directory accordingly.
- A four-byte XFP can collide. Always verify a match manually on the Coldcard before relying on it.
- Python, NumPy, JAX, and accelerator runtimes cannot guarantee that every temporary host or device buffer is securely zeroized. The CLI overwrites mutable host buffers when practical, but this remains a limitation compared with an all-Rust process.
- Run only one Rust or JAX recovery process against a state directory at a time.

## Installation

The project requires Python 3.12 or newer, uses `uv`, and includes a cross-platform lockfile.

CPU on Linux or Apple silicon:

```sh
cd jax
uv sync --locked
```

When changing the shared Rust core in a development checkout, force an extension rebuild before testing:

```sh
uv sync --locked --reinstall-package recoverme-jax
```

NVIDIA CUDA 13 on supported Linux systems:

```sh
uv sync --locked --extra cuda13
```

CUDA 12 remains available for older supported NVIDIA hardware:

```sh
uv sync --locked --extra cuda12
```

The CUDA extras are mutually exclusive. JAX does not provide a supported Apple Metal backend; macOS uses the JAX CPU backend. An explicit `--backend cuda` request fails instead of silently falling back.

## Commands

Create or validate the shared state and inspect exact duplicate-free counts:

```sh
uv run recoverme-jax plan \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --fingerprint 0123abcd \
  --state-dir recovery-state \
  --lowercase-already-tried
```

Benchmark the available JAX devices. Seed derivation, CPU fingerprint work, and complete checks are reported separately:

```sh
uv run recoverme-jax benchmark \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --backend auto
```

Run or resume through an explicit phase:

```sh
uv run recoverme-jax run \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --through written-case \
  --backend auto
```

The first Ctrl-C requests a clean stop. The current JAX batch is fully derived, fingerprinted, and atomically checkpointed before the process exits. A failed or terminated batch is replayed because its cursor was never committed.

Reject a candidate after manual Coldcard verification shows a false XFP collision:

```sh
uv run recoverme-jax reject-match \
  --state-dir recovery-state \
  --match-id CANDIDATE_ID
```

## Checkpoint compatibility

Rust and JAX share algorithm-v2 `manifest.json` and `runtime.json` files because the Python extension calls the same Rust planner and atomic state implementation. Either CLI may resume the other's completed batch. JAX performance records live in `jax-benchmarks-v1.json`; Rust ignores that sidecar.

Algorithm-v1 state is intentionally unsupported because the older planner could emit identical no-space bytes from different word segmentations. No migration is provided.

## Validation and benchmarks

```sh
uv run ruff format --check python tests
uv run ruff check python tests
uv run mypy
uv run pytest -m "not cuda"
uv run pytest -m cuda -rs
```

Set `RECOVERME_REQUIRE_CUDA=1` when CUDA validation is mandatory. CUDA tests fail clearly instead of skipping when JAX cannot find an NVIDIA device.

JAX compilation is excluded from throughput but printed separately. Seed timing waits for asynchronous device work and includes seed transfer back to the host. Complete-check timing adds the Rust CPU fingerprint stage, making a CPU secp256k1 bottleneck visible.
