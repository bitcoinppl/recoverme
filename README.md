# recoverme

`recoverme` is an offline Coldcard BIP39 passphrase recovery tool. It searches
deterministically across written words, capitalization, ordering, spacing, and
nearby BIP39 words while keeping resumable owner-only state.

> This software cannot guarantee recovery. Use it only for wallets you own or
> are explicitly authorized to recover. Work on an offline computer, never put
> a real mnemonic or passphrase in source control, and always verify a possible
> match on the Coldcard.

The Rust CLI is the stable implementation. The [JAX frontend](jax/README.md) is
experimental and intended for users who can evaluate its Python and accelerator
security tradeoffs.

## Candidate model

The simple input is one written token per line. There is no fixed seven-word
assumption; a recipe may contain up to 100 slots, though order, case, spacing,
and substitution choices grow combinatorially.

The main search controls are:

- `--order written` keeps token positions; `--order permuted` tries nearby
  swaps before every unique permutation
- `--spacing concatenated` joins tokens with no spaces
- `--spacing between` inserts one space between tokens
- `--spacing both` tries concatenated and between-token forms
- `--spacing coldcard` tries every leading-space combination produced by
  Coldcard's Add Word workflow, including a possible space before the first word
- `--concatenated-already-tried` removes the all-concatenated pattern from
  `both` or `coldcard`
- `--neighbors N` retains the N closest English BIP39 words for each token
- `--max-replacements N` creates deterministic `neighbor-N-lower` and
  `neighbor-N-case` phases up to the requested slot count
- `--lowercase-already-tried` removes every lowercase-only phase

Every displayed "Exact passphrase" is the byte string supplied to BIP39,
including any spaces.

## Build

The portable CPU build works everywhere supported by Rust:

```sh
cargo build --release
```

Optional CubeCL backends are available for supported toolchains:

```sh
cargo build --release --features metal
cargo build --release --features cuda
```

The Metal build also provides a `hybrid` backend. Benchmark on the target
machine instead of assuming an accelerator is faster.

## Protect the inputs

Put the English BIP39 mnemonic and written words in separate files, then make
them owner-only:

```sh
chmod 600 mnemonic.txt written-words.txt
```

`written-words.txt` accepts any nonempty number of ASCII-letter tokens, one per
line. Tokens need not already be valid BIP39 words.

Create the plan:

```sh
recoverme plan \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --fingerprint 0123abcd \
  --state-dir recovery-state \
  --order permuted \
  --spacing both
```

Inspect the exact counts before authorizing a phase. Then benchmark and run:

```sh
recoverme benchmark \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --backend auto \
  --autotune

recoverme run \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --through neighbor-1-case \
  --backend auto
```

Progress is committed atomically after each completed batch. Ctrl-C stops
after the current cryptographic batch; the same command resumes at the next
unverified candidate.

## Advanced recipes

An owner-only TOML recipe can express ranked alternatives and optional slots:

```toml
version = 1

[[slots]]
alternatives = ["orchard", "orange"]

[[slots]]
alternatives = ["velvet"]
optional = true
```

Alternatives are ranked in file order. Use `--recipe-file` instead of
`--words-file` on `plan`, `benchmark`, and `run`:

```sh
chmod 600 recipe.toml
recoverme plan \
  --mnemonic-file mnemonic.txt \
  --recipe-file recipe.toml \
  --fingerprint 0123abcd \
  --state-dir recovery-state
```

## Configuration and environment

An owner-only TOML file can provide defaults. Command-line values override
environment variables, which override the config file, which overrides built-in
defaults.

```toml
mnemonic_file = "mnemonic.txt"
words_file = "written-words.txt"
fingerprint = "0123abcd"
state_dir = "recovery-state"
neighbors = 3
max_replacements = 2
order = "permuted"
spacing = "coldcard"
concatenated_already_tried = true
```

```sh
chmod 600 recoverme.toml
recoverme --config recoverme.toml plan
```

Supported scoped environment variables include:

- `RECOVERME_CONFIG`
- `RECOVERME_MNEMONIC_FILE`, `RECOVERME_WORDS_FILE`, `RECOVERME_RECIPE_FILE`
- `RECOVERME_MNEMONIC`, `RECOVERME_WORDS`, `RECOVERME_FINGERPRINT`
- `RECOVERME_MASTER_XPUB_FILE`, `RECOVERME_STATE_DIR`
- `RECOVERME_NEIGHBORS`, `RECOVERME_MAX_REPLACEMENTS`
- `RECOVERME_ORDER`, `RECOVERME_SPACING`
- `RECOVERME_LOWERCASE_ALREADY_TRIED`
- `RECOVERME_CONCATENATED_ALREADY_TRIED`

Environment inputs are not printed, but another process running as the same
user may be able to inspect them. Owner-only files are preferred.

## Match verification and state

A four-byte XFP can collide. If available, provide an owner-only depth-zero
master XPUB with `--master-xpub-file` so complete public-key verification can
reject XFP collisions automatically. Manual Coldcard verification remains
required.

If manual verification rejects a pending match:

```sh
recoverme reject-match --state-dir recovery-state --match-id CANDIDATE_ID
```

The state directory contains hashes, progress, benchmarks, and pending exact
passphrases, but not the mnemonic or full candidate stream. Keep it private.
Current checkpoints use state format v2 and candidate algorithm v3. Older state
is intentionally not migrated; create a new state directory after upgrading.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Licensed under either the Apache License, Version 2.0 or the MIT License, at
your option.
