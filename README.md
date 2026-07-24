# recoverme

`recoverme` is an offline BIP39 passphrase and mnemonic recovery tool. It
searches deterministically across written passphrase clues or missing mnemonic
positions and can resume interrupted searches.

> This software cannot guarantee recovery. Work on an offline computer, never
> put a real mnemonic or passphrase in source control, and always verify a
> possible match on the Coldcard.

The Rust CLI is the stable implementation. The [JAX frontend](jax/README.md) is
experimental and intended for users who can evaluate its Python and accelerator
security tradeoffs.

Mnemonic recovery currently uses the CPU backend. Passphrase recovery supports
the compiled CPU and accelerator backends described below.

See the [changelog](CHANGELOG.md) for release details.

## Candidate model

A word list contains one written token per line. Recipes may contain up to 100
slots; the tool does not assume seven words. Order, case, spacing, and
substitution choices grow combinatorially.

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

## Install

The v0.2.0 preview provides these GitHub-built binaries:

| Platform | Backend | Archive |
| --- | --- | --- |
| Apple Silicon macOS | CPU, Metal, and hybrid | `recoverme-v0.2.0-aarch64-apple-darwin.tar.gz` |
| x86-64 Linux | Static CPU | `recoverme-v0.2.0-x86_64-unknown-linux-musl.tar.gz` |
| ARM64 Linux | Static CPU | `recoverme-v0.2.0-aarch64-unknown-linux-musl.tar.gz` |

Windows and Intel macOS binaries are not provided. Install the matching binary
with:

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://raw.githubusercontent.com/bitcoinppl/recoverme/v0.2.0/install.sh |
  sh
```

The installer verifies the archive checksum. If an authenticated GitHub CLI is
available, it also verifies the artifact attestations. The default installation
directory is `/usr/local/bin`; set `RECOVERME_INSTALL_DIR` to use another:

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://raw.githubusercontent.com/bitcoinppl/recoverme/v0.2.0/install.sh |
  RECOVERME_INSTALL_DIR="$HOME/.local/bin" sh
```

For a manual installation, download the matching archive and checksum manifest
from [GitHub Releases](https://github.com/bitcoinppl/recoverme/releases). The
following commands require the GitHub CLI:

```sh
repo=bitcoinppl/recoverme
version=v0.2.0
archive=recoverme-v0.2.0-aarch64-apple-darwin.tar.gz

gh release download "$version" \
  --repo "$repo" \
  --pattern "$archive" \
  --pattern SHA256SUMS
```

Set `archive` to the matching Linux name when needed. Verify both the checksum
and GitHub build provenance before extracting or running the binary:

```sh
if command -v sha256sum >/dev/null; then
  grep --fixed-strings " $archive" SHA256SUMS | sha256sum --check
else
  grep --fixed-strings " $archive" SHA256SUMS | shasum -a 256 --check
fi

gh attestation verify SHA256SUMS --repo "$repo"
gh attestation verify "$archive" --repo "$repo"
```

Install the verified binary:

```sh
directory=${archive%.tar.gz}
tar -xzf "$archive"
sudo install -m 0755 "$directory/recoverme" /usr/local/bin/recoverme
recoverme --version
```

The v0.2.0 macOS binary is not signed or notarized. After verification, macOS
may require selecting **Open Anyway** for `recoverme` in **System Settings →
Privacy & Security**.

## Build

The CPU source build supports Unix platforms with Rust support. Windows is
unsupported because the owner-only secret-file checks do not validate Windows
ACLs.

```sh
cargo build --release
```

Supported toolchains can also build the optional CubeCL backends:

```sh
cargo build --release --features metal
cargo build --release --features cuda
```

The Metal build also provides a fixed-share `hybrid` backend. The CUDA build
provides the GPU-only `cuda` backend. Benchmark on the target machine instead
of assuming an accelerator is faster.

### Experimental CUDA support

The Rust CubeCL and JAX CUDA backends are experimental and source-only. Neither
has been validated on NVIDIA hardware for this release, and CUDA binaries are
not included in the release downloads.

The Rust CUDA backend derives BIP39 seeds on the GPU. With a master XPUB
target, it also derives and filters BIP32 master chain codes on the GPU, copies
back only the compacted survivor indices, and performs full public-key
confirmation on the CPU. Fingerprint-only targets still require CPU
secp256k1 and HASH160 verification for every derived seed.

Tune and use the normal GPU-only path with:

```sh
recoverme benchmark \
  --state-dir recovery-state \
  --backend cuda \
  --autotune

recoverme run \
  --state-dir recovery-state \
  --through written-case \
  --backend cuda
```

The tuner measures complete checks rather than seed derivations alone. The
saved choice is bound to the CUDA device UUID, compute capability, driver
version, CPU model, and Rayon thread count; retune after hardware, driver, or
CPU-thread changes.

CUDA address verification remains a benchmark-only experiment. Moving
fingerprint or address secp256k1 verification onto CUDA requires independent
correctness testing and at least a 10% complete-check improvement over the
CPU-assisted pipeline before it can become a recovery backend.

Testing and code contributions are welcome. If you have an NVIDIA GPU, please
[open an issue](https://github.com/bitcoinppl/recoverme/issues) with the GPU
model, driver and CUDA versions, build command, test results, and benchmark
output.

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

## Recover missing mnemonic words

Mnemonic recovery supports English BIP39 mnemonics with 12, 15, 18, 21, or 24
positions. Create an owner-only template containing one exact BIP39 word or `?`
per line. The number of known and missing positions is determined by the
template:

```text
abandon
ability
able
about
above
absent
absorb
abstract
?
?
?
?
```

The final word contains both entropy and checksum bits. `recoverme` enumerates
the unknown entropy and constructs or validates the checksum automatically.

Protect the template, passphrase, and master XPUB:

```sh
chmod 600 mnemonic-template.txt passphrase.txt master-xpub.txt
```

Create a resumable plan with a known passphrase:

```sh
recoverme mnemonic plan \
  --template-file mnemonic-template.txt \
  --passphrase-file passphrase.txt \
  --master-xpub-file master-xpub.txt \
  --state-dir mnemonic-state
```

For a wallet with no BIP39 passphrase, make that choice explicit:

```sh
recoverme mnemonic plan \
  --template-file mnemonic-template.txt \
  --empty-passphrase \
  --master-xpub-file master-xpub.txt \
  --state-dir mnemonic-state
```

Benchmark or run the CPU search:

```sh
recoverme mnemonic benchmark \
  --template-file mnemonic-template.txt \
  --empty-passphrase \
  --state-dir mnemonic-state

recoverme mnemonic run \
  --template-file mnemonic-template.txt \
  --empty-passphrase \
  --state-dir mnemonic-state
```

Mnemonic recovery requires a depth-zero master XPUB. A four-byte fingerprint
is too weak to identify one mnemonic reliably in large searches. Runtime state
stores a matching candidate's deterministic rank rather than its recovered
words; the protected template is required to display it again.

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

Mnemonic-recovery defaults use the same config file format:

```toml
template_file = "mnemonic-template.txt"
empty_passphrase = true
master_xpub_file = "master-xpub.txt"
state_dir = "mnemonic-state"
```

Set `passphrase_file` instead of `empty_passphrase` when the wallet has a
BIP39 passphrase.

```sh
chmod 600 recoverme.toml
recoverme --config recoverme.toml plan
```

Supported scoped environment variables include:

- `RECOVERME_CONFIG`
- `RECOVERME_MNEMONIC_FILE`, `RECOVERME_WORDS_FILE`, `RECOVERME_RECIPE_FILE`
- `RECOVERME_TEMPLATE_FILE`, `RECOVERME_PASSPHRASE_FILE`
- `RECOVERME_EMPTY_PASSPHRASE`
- `RECOVERME_MNEMONIC`, `RECOVERME_WORDS`, `RECOVERME_FINGERPRINT`
- `RECOVERME_MASTER_XPUB_FILE`, `RECOVERME_STATE_DIR`
- `RECOVERME_NEIGHBORS`, `RECOVERME_MAX_REPLACEMENTS`
- `RECOVERME_ORDER`, `RECOVERME_SPACING`
- `RECOVERME_LOWERCASE_ALREADY_TRIED`
- `RECOVERME_CONCATENATED_ALREADY_TRIED`

`recoverme` does not print environment inputs, but another process running as
the same user may be able to inspect them. Prefer owner-only files.

## Match verification and state

A four-byte XFP can collide. Use `--master-xpub-file` to provide an owner-only
depth-zero master XPUB when available. `recoverme` can then reject XFP
collisions through complete public-key verification. Manual Coldcard
verification remains required.

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
cargo check --all-targets --features cuda

# requires an NVIDIA GPU
cargo test --features cuda cuda_ -- --ignored
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Licensed under either the Apache License, Version 2.0 or the MIT License, at
your option.
