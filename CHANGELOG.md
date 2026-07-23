# Changelog

## [0.2.0] - 2026-07-23

`recoverme` 0.2.0 is an initial preview release of the offline Coldcard BIP39
passphrase recovery tool.

### Highlights

- Searches deterministically across written words, capitalization, ordering,
  spacing, and nearby BIP39 words
- Commits owner-only progress atomically so interrupted searches can resume at
  the next unverified candidate
- Supports simple word lists and ranked recipes with alternatives and optional
  slots
- Provides a portable CPU backend and Metal and hybrid backends on Apple
  Silicon
- Installs checksummed, attested GitHub-built binaries on supported macOS and
  Linux systems with a single shell script
- Verifies the complete depth-zero master XPUB when supplied, avoiding reliance
  on a potentially colliding four-byte fingerprint
- Includes an experimental, source-only JAX frontend for users who can evaluate
  its Python and accelerator security tradeoffs

### Performance

An M4 Pro measured approximately 93,000 complete master-XPUB checks per second.
This result is machine- and workload-specific; benchmark the intended recovery
plan on the offline computer that will run it.

### Preview notice

This software has not been audited and cannot guarantee recovery. Use it only
for wallets you own or are explicitly authorized to recover, keep all recovery
material offline, and verify every possible match on the Coldcard.
