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

## Pull requests

Keep changes focused, explain any candidate-order or state-format impact, and
include the commands used for verification. Documentation and examples must
contain only generated public data.

Contributions are accepted under the project's MIT OR Apache-2.0 license.
