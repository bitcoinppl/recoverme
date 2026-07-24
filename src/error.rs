use std::{io, path::PathBuf};

use thiserror::Error;

/// Errors produced by recovery planning and execution
#[derive(Debug, Error)]
pub enum RecoverError {
    /// Input or state file could not be read or written
    #[error("I/O error for {path}: {source}")]
    Io {
        /// Affected path
        path: PathBuf,
        /// Underlying I/O error
        source: io::Error,
    },
    /// A protected file grants access to group or other users
    #[error("protected file must be owner-only (mode 0600 or stricter): {0}")]
    InsecurePermissions(PathBuf),
    /// A required environment-backed recovery input is absent
    #[error("required environment variable {0} is not set")]
    MissingEnvironmentVariable(&'static str),
    /// An environment-backed recovery input is not valid UTF-8
    #[error("environment variable {0} must contain valid UTF-8")]
    InvalidEnvironmentVariable(&'static str),
    /// Mnemonic input is not a valid English BIP39 mnemonic
    #[error("invalid English BIP39 mnemonic: {0}")]
    InvalidMnemonic(String),
    /// Fingerprint input is not exactly eight hexadecimal digits
    #[error("fingerprint must be exactly eight hexadecimal digits: {0}")]
    InvalidFingerprint(String),
    /// Master extended public key is invalid or is not depth zero
    #[error("invalid master extended public key: {0}")]
    InvalidMasterXpub(String),
    /// Master extended public key and requested fingerprint disagree
    #[error("master extended public key does not match the requested fingerprint")]
    MasterXpubFingerprintMismatch,
    /// No written words were supplied
    #[error("written-words file must contain at least one word")]
    NoWrittenWords,
    /// A written token contains unsupported characters
    #[error("written word {line} must contain ASCII letters only")]
    InvalidWrittenWord { line: usize },
    /// Search configuration is outside supported bounds
    #[error("invalid recovery setting: {0}")]
    InvalidSetting(String),
    /// Candidate counting overflowed the durable counter type
    #[error("candidate count exceeds u128")]
    CountOverflow,
    /// State JSON is invalid
    #[error("invalid state file {path}: {source}")]
    InvalidState {
        /// Affected path
        path: PathBuf,
        /// JSON parser error
        source: serde_json::Error,
    },
    /// Existing state belongs to different secret inputs or settings
    #[error("state directory belongs to different recovery inputs or settings")]
    StateMismatch,
    /// State schema or algorithm version is unsupported
    #[error("unsupported state version {0}")]
    UnsupportedStateVersion(u32),
    /// Requested phase is not enabled by the configured replacement limit
    #[error("phase {0} is disabled by max-replacements")]
    DisabledPhase(String),
    /// Backend was not compiled into this binary or cannot run here
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    /// Seed derivation failed
    #[error("seed derivation failed: {0}")]
    SeedDerivation(String),
    /// A benchmark plan cannot fill every warmup and timed batch
    #[error(
        "benchmark requires {required} candidates but the selected recovery phases contain {available}"
    )]
    InsufficientBenchmarkCandidates {
        /// Number of candidates needed for the requested measurement
        required: usize,
        /// Number of candidates available in the selected recovery phases
        available: u128,
    },
    /// Candidate preparation thread terminated unexpectedly
    #[error("candidate preparation worker terminated unexpectedly")]
    CandidatePreparationPanic,
    /// Fingerprint derivation failed
    #[error("fingerprint derivation failed: {0}")]
    FingerprintDerivation(String),
    /// Search cannot continue while candidates await manual verification
    #[error("pending XFP matches must be verified or rejected before resuming")]
    PendingMatches,
    /// Requested match identifier does not exist or is already resolved
    #[error("pending match not found: {0}")]
    MatchNotFound(String),
}

impl RecoverError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
