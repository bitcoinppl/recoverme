# Contributing

Contributions are welcome for correctness, performance, documentation, and
additional deterministic recovery strategies.

## Safety

- Never commit or share real wallet material
- Use the public fixtures under `src` and `jax/tests`
- Keep secret and state files owner-only during manual testing
- Preserve deterministic candidate ordering and checkpoint compatibility, or
  deliberately bump the corresponding format version
- Add tests for user-visible behavior and non-obvious candidate invariants

## Rust checks

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Run feature-specific checks when changing accelerator code.

## JAX checks

```sh
cd jax
uv sync --locked --reinstall-package recoverme-jax
uv run ruff format --check python tests
uv run ruff check python tests
uv run mypy
uv run pytest -m "not cuda"
```

CUDA behavior should be tested on supported NVIDIA hardware. Set
`RECOVERME_REQUIRE_CUDA=1` when a missing CUDA device must fail the suite.

## GPUQ CUDA canary

Build and push the CUDA canary:

```sh
tests/cuda-canary/build-image.sh ghcr.io/bitcoinppl/recoverme-cuda-canary:<tag>
```

The script uses Docker Buildx when available, otherwise it falls back to
Namespace's `nsc` remote builder. It requires `skopeo` to resolve the pushed
image digest. After the push, copy the digest-qualified image reference printed
by the script into `gpuq.toml`.

## Pull requests

Keep changes focused, explain any candidate-order or state-format impact, and
include the commands used for verification. Documentation and examples must
contain only generated public data.

## Releases

Keep the versions in `Cargo.toml`, `jax/pyproject.toml`, and
`jax/native/Cargo.toml` aligned, and add a dated changelog entry before starting
a release.

Run the Release workflow manually against the intended revision first. Inspect
the complete workflow artifact, verify all checksums and attestations, and test
the binaries on their target platforms. After the dry run passes, create and
push an annotated `vX.Y.Z` tag. The tag workflow creates a draft prerelease; it
does not publish automatically. Review the draft and its assets before
publishing it from GitHub. Enable immutable releases before publishing the first
release.

Contributions are accepted under the project's MIT OR Apache-2.0 license.
