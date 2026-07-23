# recoverme-jax

`recoverme-jax` is an experimental JAX frontend for recoverme. It runs
PBKDF2-HMAC-SHA512 in fixed-size JIT-compiled batches while using the Rust v3
candidate planner and v2 checkpoint implementation.

Use the stable Rust CLI unless measured JAX performance justifies the larger
Python, NumPy, JAX, and accelerator-runtime attack surface. Temporary host and
device buffers cannot be guaranteed to be securely zeroized.

## Install

Python 3.12 or newer and `uv` are required:

```sh
cd jax
uv sync --locked
```

After changing the shared Rust core:

```sh
uv sync --locked --reinstall-package recoverme-jax
```

CUDA support is experimental and has not been validated on NVIDIA hardware for
this release. On NVIDIA Linux systems, add either `--extra cuda12` or
`--extra cuda13`. The extras are mutually exclusive. Compare CUDA results with
the CPU backend before using it for a recovery. JAX has no supported Apple Metal
backend; macOS uses JAX CPU.

Testing and code contributions are welcome. Please
[open an issue](https://github.com/bitcoinppl/recoverme/issues) with the GPU
model, driver and CUDA versions, test results, and benchmark output.

## Use

Input files must be owner-only:

```sh
chmod 600 mnemonic.txt written-words.txt
```

Create a shared recovery plan with the same recipe, ordering, spacing, and
substitution controls as Rust:

```sh
uv run recoverme-jax plan \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --fingerprint 0123abcd \
  --state-dir recovery-state \
  --order permuted \
  --spacing coldcard \
  --concatenated-already-tried
```

Use `--recipe-file recipe.toml` instead of `--words-file` for an advanced v1
recipe. The same input choice must be supplied to later commands.

```sh
uv run recoverme-jax benchmark \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --backend auto

uv run recoverme-jax run \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --through neighbor-3-case \
  --backend auto
```

The first Ctrl-C requests a clean stop after committing the current batch. A
failed or terminated batch is replayed because its cursor was never committed.

A four-byte XFP can collide. Verify every pending match on the Coldcard. Reject
a false collision with:

```sh
uv run recoverme-jax reject-match \
  --state-dir recovery-state \
  --match-id CANDIDATE_ID
```

## Shared-state compatibility

Rust and JAX share algorithm-v3 state. Either CLI may resume the other's
completed batch when the recipe and settings match exactly. JAX performance
records remain in the separate `jax-benchmarks-v1.json` sidecar.

## Validate

```sh
uv run ruff format --check python tests
uv run ruff check python tests
uv run mypy
uv run pytest -m "not cuda"
```

Set `RECOVERME_REQUIRE_CUDA=1` when CUDA validation is mandatory.
