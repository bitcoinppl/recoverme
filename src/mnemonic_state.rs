use std::path::{Path, PathBuf};

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use crate::{
    domain::{BackendKind, CandidateId, MasterXpubTarget, TargetFingerprint, VerificationTarget},
    error::RecoverError,
    mnemonic::{MnemonicCandidate, MnemonicCursor, MnemonicPlan},
    state::{
        atomic_write_json, check_private_state, create_private_directory, protect_existing_file,
        read_json, unix_seconds, BenchmarkRecord, MatchStatus,
    },
};

const MANIFEST_FILE: &str = "manifest.json";
const RUNTIME_FILE: &str = "runtime.json";
const STATE_FORMAT_VERSION: u32 = 1;
const ALGORITHM_VERSION: u32 = 1;

/// Immutable metadata for a mnemonic recovery
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MnemonicRecoveryManifest {
    /// State document format discriminator
    pub state_format_version: u32,
    /// Mnemonic candidate-ordering version
    pub algorithm_version: u32,
    /// Recovery kind discriminator
    pub recovery_kind: String,
    /// SHA-256 identifier of secret inputs and target
    pub spec_hash: String,
    /// Target master fingerprint
    pub target_fingerprint: TargetFingerprint,
    /// Strong wallet identity required for mnemonic recovery
    pub master_xpub: MasterXpubTarget,
    /// Configured BIP39 word count
    pub word_count: usize,
    /// Number of known positions
    pub known_word_count: usize,
    /// Number of unknown entropy bits
    pub unknown_entropy_bits: usize,
    /// Exact number of entropy assignments, encoded as decimal
    pub total_work: String,
    /// Manifest creation time as Unix seconds
    pub created_at_unix: u64,
}

/// A mnemonic match stored by rank without persisting recovered words
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnemonicMatchRecord {
    /// Stable candidate identifier
    pub id: CandidateId,
    /// Entropy-assignment rank encoded as decimal
    pub rank: String,
    /// Current manual-verification status
    pub status: MatchStatus,
}

impl MnemonicMatchRecord {
    /// Construct a pending record from a verified candidate
    pub fn pending(candidate: &MnemonicCandidate) -> Self {
        Self {
            id: candidate.id().clone(),
            rank: candidate.rank().to_str_radix(10),
            status: MatchStatus::Pending,
        }
    }

    /// Parse the durable candidate rank
    pub fn parsed_rank(&self) -> Result<BigUint, RecoverError> {
        BigUint::parse_bytes(self.rank.as_bytes(), 10).ok_or_else(|| {
            RecoverError::InvalidSetting("mnemonic match contains an invalid rank".into())
        })
    }
}

/// Mutable mnemonic-recovery progress
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MnemonicRuntimeState {
    /// Next entropy assignment
    pub cursor: MnemonicCursor,
    /// CPU benchmark measurements
    pub benchmarks: Vec<BenchmarkRecord>,
    /// Strong wallet-identity matches
    pub matches: Vec<MnemonicMatchRecord>,
}

/// Durable state for a mnemonic recovery
pub struct MnemonicRecoveryState {
    directory: PathBuf,
    manifest: MnemonicRecoveryManifest,
    runtime: MnemonicRuntimeState,
}

impl MnemonicRecoveryState {
    /// Open matching state or create a new owner-only state directory
    pub fn open_or_create(
        directory: &Path,
        spec_hash: String,
        target: VerificationTarget,
        plan: &MnemonicPlan,
    ) -> Result<Self, RecoverError> {
        let Some(master_xpub) = target.master_xpub().cloned() else {
            return Err(RecoverError::InvalidSetting(
                "mnemonic recovery requires a master XPUB target".into(),
            ));
        };
        create_private_directory(directory)?;
        let manifest_path = directory.join(MANIFEST_FILE);
        let runtime_path = directory.join(RUNTIME_FILE);
        protect_existing_file(&manifest_path)?;
        protect_existing_file(&runtime_path)?;

        let expected = MnemonicRecoveryManifest {
            state_format_version: STATE_FORMAT_VERSION,
            algorithm_version: ALGORITHM_VERSION,
            recovery_kind: "mnemonic".into(),
            spec_hash,
            target_fingerprint: target.fingerprint(),
            master_xpub,
            word_count: plan.template().word_count(),
            known_word_count: plan.template().known_word_count(),
            unknown_entropy_bits: plan.unknown_entropy_bits(),
            total_work: plan.total_work().to_str_radix(10),
            created_at_unix: unix_seconds(),
        };
        let manifest = if manifest_path.exists() {
            let existing: MnemonicRecoveryManifest = read_json(&manifest_path)?;
            validate_manifest(&existing, &expected)?;
            existing
        } else {
            atomic_write_json(&manifest_path, &expected)?;
            expected
        };
        let runtime = if runtime_path.exists() {
            read_json(&runtime_path)?
        } else {
            let runtime = MnemonicRuntimeState::default();
            atomic_write_json(&runtime_path, &runtime)?;
            runtime
        };

        Ok(Self {
            directory: directory.to_owned(),
            manifest,
            runtime,
        })
    }

    /// Open existing mnemonic state without loading secret inputs
    pub fn open_existing(directory: &Path) -> Result<Self, RecoverError> {
        check_private_state(directory)?;
        let manifest: MnemonicRecoveryManifest = read_json(&directory.join(MANIFEST_FILE))?;
        if manifest.state_format_version != STATE_FORMAT_VERSION {
            return Err(RecoverError::UnsupportedStateVersion(
                manifest.state_format_version,
            ));
        }
        if manifest.algorithm_version != ALGORITHM_VERSION {
            return Err(RecoverError::UnsupportedStateVersion(
                manifest.algorithm_version,
            ));
        }
        if manifest.recovery_kind != "mnemonic" {
            return Err(RecoverError::StateMismatch);
        }
        let runtime = read_json(&directory.join(RUNTIME_FILE))?;
        Ok(Self {
            directory: directory.to_owned(),
            manifest,
            runtime,
        })
    }

    /// Immutable recovery manifest
    pub fn manifest(&self) -> &MnemonicRecoveryManifest {
        &self.manifest
    }

    /// Strong wallet identity stored in the manifest
    pub fn verification_target(&self) -> VerificationTarget {
        VerificationTarget::from_master_xpub(self.manifest.master_xpub.clone())
    }

    /// Mutable runtime state
    pub fn runtime(&self) -> &MnemonicRuntimeState {
        &self.runtime
    }

    /// Clone the next entropy-assignment cursor
    pub fn cursor(&self) -> MnemonicCursor {
        self.runtime.cursor.clone()
    }

    /// Whether a verified candidate awaits manual confirmation
    pub fn has_pending_matches(&self) -> bool {
        self.runtime
            .matches
            .iter()
            .any(|record| record.status == MatchStatus::Pending)
    }

    /// Atomically persist a completed batch and its matches
    pub fn complete_chunk(
        &mut self,
        cursor: MnemonicCursor,
        matches: Vec<MnemonicMatchRecord>,
    ) -> Result<(), RecoverError> {
        self.runtime.cursor = cursor;
        for record in matches {
            if !self
                .runtime
                .matches
                .iter()
                .any(|known| known.id == record.id)
            {
                self.runtime.matches.push(record);
            }
        }
        self.save_runtime()
    }

    /// Replace the stored CPU benchmark
    pub fn record_benchmark(&mut self, benchmark: BenchmarkRecord) -> Result<(), RecoverError> {
        if benchmark.backend != BackendKind::Cpu {
            return Err(RecoverError::BackendUnavailable(
                "mnemonic recovery currently supports only the CPU backend".into(),
            ));
        }
        self.runtime.benchmarks.clear();
        self.runtime.benchmarks.push(benchmark);
        self.save_runtime()
    }

    /// Most recent CPU benchmark
    pub fn latest_benchmark(&self) -> Option<&BenchmarkRecord> {
        self.runtime.benchmarks.last()
    }

    /// Mark a pending mnemonic match as rejected
    pub fn reject_match(&mut self, id: &str) -> Result<(), RecoverError> {
        let Some(record) = self
            .runtime
            .matches
            .iter_mut()
            .find(|record| record.id.0 == id && record.status == MatchStatus::Pending)
        else {
            return Err(RecoverError::MatchNotFound(id.to_owned()));
        };
        record.status = MatchStatus::Rejected;
        self.save_runtime()
    }

    fn save_runtime(&self) -> Result<(), RecoverError> {
        atomic_write_json(&self.directory.join(RUNTIME_FILE), &self.runtime)
    }
}

fn validate_manifest(
    existing: &MnemonicRecoveryManifest,
    expected: &MnemonicRecoveryManifest,
) -> Result<(), RecoverError> {
    if existing.state_format_version != STATE_FORMAT_VERSION {
        return Err(RecoverError::UnsupportedStateVersion(
            existing.state_format_version,
        ));
    }
    if existing.algorithm_version != ALGORITHM_VERSION {
        return Err(RecoverError::UnsupportedStateVersion(
            existing.algorithm_version,
        ));
    }
    if existing.recovery_kind != expected.recovery_kind
        || existing.spec_hash != expected.spec_hash
        || existing.target_fingerprint != expected.target_fingerprint
        || existing.master_xpub != expected.master_xpub
        || existing.word_count != expected.word_count
        || existing.known_word_count != expected.known_word_count
        || existing.unknown_entropy_bits != expected.unknown_entropy_bits
        || existing.total_work != expected.total_work
    {
        return Err(RecoverError::StateMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bip32::{Prefix, XPrv};
    use bip39::{Language, Mnemonic};
    use tempfile::tempdir;

    use super::*;
    use crate::mnemonic::MnemonicTemplate;

    fn target() -> VerificationTarget {
        let mnemonic = Mnemonic::parse_in(
            Language::English,
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
        )
        .unwrap();
        let xpub = XPrv::new(mnemonic.to_seed("")).unwrap().public_key();
        let serialized = xpub.to_string(Prefix::XPUB);
        VerificationTarget::from_master_xpub(MasterXpubTarget::parse(&serialized).unwrap())
    }

    #[test]
    fn cursor_and_match_rank_round_trip_without_plaintext_mnemonic() {
        let directory = tempdir().unwrap();
        let template = MnemonicTemplate::parse(
            "abandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\n?",
        )
        .unwrap();
        let plan = MnemonicPlan::compile(template).unwrap();
        let candidate = plan.candidate_at(&BigUint::default()).unwrap().unwrap();
        let mut state =
            MnemonicRecoveryState::open_or_create(directory.path(), "spec".into(), target(), &plan)
                .unwrap();
        let cursor = MnemonicCursor {
            next_rank: BigUint::from(4_u8),
        };
        state
            .complete_chunk(cursor, vec![MnemonicMatchRecord::pending(&candidate)])
            .unwrap();

        let reopened = MnemonicRecoveryState::open_existing(directory.path()).unwrap();
        assert_eq!(reopened.cursor().next_rank, BigUint::from(4_u8));
        assert_eq!(
            reopened.runtime().matches[0].parsed_rank().unwrap(),
            BigUint::default()
        );
        let runtime = std::fs::read_to_string(directory.path().join(RUNTIME_FILE)).unwrap();
        assert!(!runtime.contains("abandon"));
    }
}
