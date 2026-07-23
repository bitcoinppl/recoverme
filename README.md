# recoverme

`recoverme` searches for a Coldcard BIP39 passphrase assembled from words that may have been reordered, capitalized differently, or written down as a nearby BIP39 word.

Every candidate is passed to BIP39 with no spaces. Spaces are used only when displaying a candidate in a readable form.

## Build

Use the CPU implementation everywhere:

```sh
cargo build --release
```

Enable the CubeCL Metal backend on macOS:

```sh
cargo build --release --features metal
```

This build also enables a `hybrid` backend that divides each batch between the
ARM CPU and Metal. Use the benchmark command on the target Mac instead of
assuming that hybrid execution is faster under every thermal or power state.

Enable the CubeCL CUDA backend on a machine with a supported NVIDIA toolchain:

```sh
cargo build --release --features cuda
```

## Inputs

Recovery inputs can come from protected files or environment variables. For
files, put the mnemonic in one file and the written passphrase words in another
file, one word per line. Both files must be readable only by their owner:

```sh
chmod 600 mnemonic.txt written-words.txt
```

The mnemonic must be valid English BIP39. A written passphrase word does not need to be in the BIP39 list because a handwriting error may be the reason recovery is needed.

When both file flags are omitted, the CLI reads:

- `SEED`: the complete BIP39 mnemonic
- `PASSPHRASE`: whitespace-separated written passphrase words
- `XFP`: the target eight-digit Coldcard fingerprint when `--fingerprint` is omitted

For example, with those variables supplied by direnv:

```sh
recoverme plan --state-dir recovery-state
recoverme benchmark --state-dir recovery-state --backend auto --autotune
recoverme run --state-dir recovery-state --through written-case --backend auto
```

`--mnemonic-file` and `--words-file` must be supplied together. Explicit
`--fingerprint` overrides `XFP`. Environment values are never printed, but
process environments may be visible to other software running as the same
user.

## Plan and benchmark

Create a protected, resumable state directory and inspect exact candidate counts:

```sh
recoverme plan \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --fingerprint 0123abcd \
  --state-dir recovery-state \
  --lowercase-already-tried
```

If the depth-zero master XPUB is available, put it in an owner-only file and
include it in the plan:

```sh
chmod 600 master-xpub.txt
recoverme plan \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --fingerprint 0123abcd \
  --master-xpub-file master-xpub.txt \
  --state-dir recovery-state
```

The XPUB must be the depth-zero master key and its fingerprint must equal
`--fingerprint`. With an accelerator backend, recovery compares the BIP32 chain
code on the device and transfers only possible matches for complete public-key
confirmation. This avoids bulk CPU secp256k1 work and makes an accidental
four-byte fingerprint collision insufficient to stop the search. Manual
Coldcard verification is still required.

Use `--lowercase-already-tried` only when an earlier search completed all lowercase-only combinations. It removes those candidates entirely instead of merely changing their priority.

Benchmark every backend compiled into the binary:

```sh
recoverme benchmark \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state
```

For a new machine or after changing its power or cooling configuration, sweep
production batch and workgroup sizes and persist the fastest result:

```sh
recoverme benchmark \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --backend auto \
  --autotune
```

The report separates raw BIP39 seed derivation, complete wallet verification,
host candidate generation, and sustained checks with next-batch preparation
overlapped with the current cryptographic batch. `run --backend auto` uses the
fastest persisted sustained result and its selected batch/workgroup sizes.

With `--lowercase-already-tried`, the ranked search starts with the words as written using Title and UPPER variants, then considers one and two nearest-word substitutions. Without it, the corresponding lowercase-only phases are included too. Nearby adjacent swaps are emitted before exhaustive unique permutations. The plan output shows the exact count, estimated duration, and four-byte fingerprint collision probability for each phase.

## Run and resume

Run only through the capitalization variants of the written words:

```sh
recoverme run \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --through written-case \
  --backend auto
```

Later, extend the same checkpoint through the one-word neighbor phase:

```sh
recoverme run \
  --mnemonic-file mnemonic.txt \
  --words-file written-words.txt \
  --state-dir recovery-state \
  --through neighbor-1-case \
  --backend auto
```

Progress is committed atomically after each completed batch. Candidate
generation for the next batch overlaps current-batch verification, but the
prefetched cursor is never committed early. Pressing Ctrl-C stops after the
current cryptographic batch, and the same command resumes at the exact next
unverified candidate.

An XFP match stops the search for manual verification on the Coldcard. A four-byte fingerprint can collide, so a match is not proof by itself. If a candidate is rejected after manual verification, mark it rejected and resume:

```sh
recoverme reject-match --state-dir recovery-state --match-id CANDIDATE_ID
```

The state directory contains hashes, progress, benchmark results, and any matching passphrase candidates. It does not store the mnemonic or the full candidate stream. Keep the directory private because a match record contains the exact passphrase.

Current checkpoints use state format v1 and candidate algorithm v2, which
assigns every distinct no-space byte string to its earliest ranked derivation.
Unversioned state and algorithm-v1 checkpoints are intentionally unsupported;
create a new state directory after upgrading. A JAX native extension must be
rebuilt against this Rust revision before sharing a newly created state
directory.

## Performance notes

The CPU backend uses parallel PBKDF2-HMAC-SHA512 with the `sha2` assembly
implementation. Metal batches use precomputed HMAC states, persistent device
constants, a rolling 16-word SHA-512 schedule, and a specialized fixed-size
PBKDF2 iteration path. The remaining work cannot be replaced by a direct XFP
comparison: every candidate still requires all 2,048 BIP39 PBKDF2 rounds. A
master XPUB removes bulk public-key derivation and collision stops, but it does
not reduce that password-stretching cost.
