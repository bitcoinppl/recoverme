# Security policy

## Sensitive recovery material

Never submit a real mnemonic, passphrase, XPUB, recovery recipe, state
directory, screenshot, log, or environment file in an issue, pull request, or
test fixture. Use generated public BIP39 vectors only.

If sensitive wallet material is exposed, assume it is compromised. Move any
funds to a newly generated wallet before continuing the report. Deleting a file
or repository does not revoke copies that may already exist.

## Reporting a vulnerability

Use the repository host's private security-advisory channel when available.
Include the affected revision, impact, and a reproduction built from public
test vectors. Do not open a public issue containing exploit details or secrets.

If no private reporting channel is configured, open a minimal public issue that
contains no vulnerability details or sensitive data and asks the maintainer to
establish private contact.

## Supported versions

Security fixes are made on the latest released version. Recovery checkpoints
are versioned and may intentionally require a new state directory after a
security-sensitive candidate-model change.

## Operational limits

This project is not a custody service and provides no recovery guarantee. Run
it offline on a computer you control, protect all inputs and state, and verify
every result on the hardware wallet.
