# Changelog

## Unreleased

### CUDA

- Add GPU-first CUDA recovery with CPU public-key confirmation
- Compact master-XPUB chain-code survivors on the GPU before host readback
- Bind tuned configurations to the CUDA device, driver, CPU model, and thread
  configuration

## [0.2.0] - 2026-07-23

`recoverme` 0.2.0 is the first preview release of the offline BIP39 passphrase
recovery tool.

### Highlights

- Searches deterministically across written words, capitalization, ordering,
  spacing, and nearby BIP39 words
- Writes progress atomically to owner-only state so interrupted searches resume
  at the next unverified candidate
- Supports simple word lists and ranked recipes with alternatives and optional
  slots
- Provides a portable CPU backend, plus Metal and hybrid backends on Apple
  Silicon
- Installs checksummed, attested GitHub-built binaries on supported macOS and
  Linux systems with one shell script
- Verifies the complete depth-zero master XPUB when supplied instead of relying
  only on a potentially colliding four-byte fingerprint
- Includes an experimental, source-only JAX frontend for users who can evaluate
  its Python and accelerator security tradeoffs
- Keeps the Rust CubeCL and JAX CUDA backends experimental and source-only;
  neither has been validated on NVIDIA hardware for this release

### Performance

In one M4 Pro benchmark, `recoverme` completed approximately 93,000 complete
master-XPUB checks per second. Results vary by machine and workload. Benchmark
the intended recovery plan on the offline computer that will run it.

### Preview notice

This software has not been audited and cannot guarantee recovery. Keep all
recovery material offline and verify every possible match on the Coldcard.
