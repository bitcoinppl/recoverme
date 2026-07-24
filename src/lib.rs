//! Deterministic, resumable BIP39 passphrase recovery

pub mod backend;
pub mod benchmark;
pub mod config;
pub mod crypto;
#[cfg(any(
    feature = "cube-cpu",
    all(feature = "metal", target_os = "macos"),
    feature = "cuda"
))]
pub mod cube_backend;
pub mod domain;
pub mod engine;
pub mod error;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod hybrid_backend;
pub mod input;
pub mod search;
pub mod state;

pub use domain::{
    BackendConfiguration, BackendKind, BatchSize, CandidateBatch, CandidateCursor, CpuShare,
    MasterXpubTarget, OrderMode, RecoveryRecipe, RecoverySettings, SearchPhase, SecretMnemonic,
    SpacingMode, TargetFingerprint, TokenSlot, VerificationTarget, WorkgroupSize, WrittenWords,
};
pub use error::RecoverError;
