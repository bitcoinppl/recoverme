use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    domain::{
        BackendConfiguration, BackendKind, Candidate, CandidateCursor, CandidateId,
        MasterXpubTarget, PhaseSummary, RecoverySettings, SearchPhase, TargetFingerprint,
        VerificationTarget, ALGORITHM_VERSION,
    },
    error::RecoverError,
};

const MANIFEST_FILE: &str = "manifest.json";
const RUNTIME_FILE: &str = "runtime.json";
const STATE_FORMAT_VERSION: u32 = 2;

/// Immutable recovery metadata stored without plaintext secrets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryManifest {
    /// State document format discriminator
    #[serde(default)]
    pub state_format_version: u32,
    /// State and candidate-ordering version
    pub algorithm_version: u32,
    /// SHA-256 identifier of the secret inputs and immutable settings
    pub spec_hash: String,
    /// Target Coldcard fingerprint
    pub target_fingerprint: TargetFingerprint,
    /// Optional master public key used for chain-code filtering
    #[serde(default)]
    pub master_xpub: Option<MasterXpubTarget>,
    /// Immutable search settings
    pub settings: RecoverySettings,
    /// Exact counts for enabled phases
    pub phases: Vec<PhaseSummary>,
    /// Manifest creation time as Unix seconds
    pub created_at_unix: u64,
}

/// Measured backend throughput used to estimate search duration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    /// Backend measured
    pub backend: BackendKind,
    /// Number of candidates in the benchmark sample
    pub candidates: usize,
    /// Batch size selected for full recovery runs
    pub batch_size: usize,
    /// Accelerator workgroup size when applicable
    pub workgroup_size: Option<u32>,
    /// CPU percentage used by an explicit hybrid backend
    #[serde(default)]
    pub cpu_share_percent: Option<u8>,
    /// Hardware identity for which this measurement is valid
    #[serde(default)]
    pub hardware_signature: Option<String>,
    /// BIP39 seed derivations per second
    pub seeds_per_second: f64,
    /// Sustained checks per second with candidate preparation overlapped
    pub checks_per_second: f64,
    /// Complete backend verifications per second without candidate generation
    #[serde(default)]
    pub verification_per_second: f64,
    /// Host candidate preparation per second
    #[serde(default)]
    pub generation_per_second: f64,
    /// Device/runtime description
    pub device: String,
    /// Measurement time as Unix seconds
    pub measured_at_unix: u64,
}

impl BenchmarkRecord {
    /// Reconstruct and validate the selected runtime configuration
    pub fn configuration(&self) -> Result<BackendConfiguration, RecoverError> {
        match self.backend {
            BackendKind::Cpu => BackendConfiguration::cpu(self.batch_size),
            BackendKind::CubeCpu | BackendKind::Metal | BackendKind::Cuda => {
                BackendConfiguration::cube(
                    self.batch_size,
                    self.workgroup_size.ok_or_else(|| {
                        RecoverError::InvalidSetting(format!(
                            "{} benchmark is missing a workgroup size",
                            self.backend
                        ))
                    })?,
                )
            }
            BackendKind::Hybrid => BackendConfiguration::hybrid(
                self.batch_size,
                self.workgroup_size.ok_or_else(|| {
                    RecoverError::InvalidSetting(format!(
                        "{} benchmark is missing a workgroup size",
                        self.backend
                    ))
                })?,
                self.cpu_share_percent.unwrap_or(35),
            ),
            BackendKind::Auto => Err(RecoverError::InvalidSetting(
                "auto cannot be stored as a benchmark backend".into(),
            )),
        }
    }
}

/// Manual disposition of a wallet-identity match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MatchStatus {
    /// Awaiting manual verification on the Coldcard
    Pending,
    /// Rejected as a random four-byte XFP collision
    Rejected,
}

/// Candidate whose configured wallet identity matched the target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchRecord {
    /// Stable candidate identifier
    pub id: CandidateId,
    /// Phase that produced the candidate
    pub phase: SearchPhase,
    /// Exact no-space passphrase
    pub passphrase: String,
    /// Readable word segmentation
    pub words: Vec<String>,
    /// Current manual-verification status
    pub status: MatchStatus,
}

impl MatchRecord {
    /// Construct a pending match from a tested candidate
    pub fn pending(candidate: &Candidate) -> Self {
        Self {
            id: candidate.id().clone(),
            phase: candidate.phase(),
            passphrase: candidate.passphrase().to_owned(),
            words: candidate.words().to_vec(),
            status: MatchStatus::Pending,
        }
    }
}

/// Mutable, atomically persisted recovery progress
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeState {
    /// Cursor pointing to the next unverified candidate
    pub cursor: CandidateCursor,
    /// Recorded backend measurements
    pub benchmarks: Vec<BenchmarkRecord>,
    /// XFP matches and their manual status
    pub matches: Vec<MatchRecord>,
}

/// Durable manifest and runtime state rooted in one protected directory
pub struct RecoveryState {
    directory: PathBuf,
    manifest: RecoveryManifest,
    runtime: RuntimeState,
}

impl RecoveryState {
    /// Open matching state or create a new protected state directory
    pub fn open_or_create(
        directory: &Path,
        spec_hash: String,
        target_fingerprint: TargetFingerprint,
        settings: RecoverySettings,
        phases: Vec<PhaseSummary>,
    ) -> Result<Self, RecoverError> {
        Self::open_or_create_for_target(
            directory,
            spec_hash,
            VerificationTarget::Fingerprint(target_fingerprint),
            settings,
            phases,
        )
    }

    /// Open matching state or create it for a typed verification target
    pub fn open_or_create_for_target(
        directory: &Path,
        spec_hash: String,
        target: VerificationTarget,
        settings: RecoverySettings,
        phases: Vec<PhaseSummary>,
    ) -> Result<Self, RecoverError> {
        create_private_directory(directory)?;
        let manifest_path = directory.join(MANIFEST_FILE);
        let runtime_path = directory.join(RUNTIME_FILE);
        protect_existing_file(&manifest_path)?;
        protect_existing_file(&runtime_path)?;

        let expected = RecoveryManifest {
            state_format_version: STATE_FORMAT_VERSION,
            algorithm_version: ALGORITHM_VERSION,
            spec_hash,
            target_fingerprint: target.fingerprint(),
            master_xpub: target.master_xpub().cloned(),
            settings,
            phases,
            created_at_unix: unix_seconds(),
        };

        let manifest = if manifest_path.exists() {
            let existing: RecoveryManifest = read_json(&manifest_path)?;
            validate_manifest(&existing, &expected)?;
            existing
        } else {
            atomic_write_json(&manifest_path, &expected)?;
            expected
        };

        let runtime = if runtime_path.exists() {
            read_json(&runtime_path)?
        } else {
            let runtime = RuntimeState::default();
            atomic_write_json(&runtime_path, &runtime)?;
            runtime
        };

        Ok(Self {
            directory: directory.to_owned(),
            manifest,
            runtime,
        })
    }

    /// Open existing state without secret inputs
    pub fn open_existing(directory: &Path) -> Result<Self, RecoverError> {
        check_private_state(directory)?;
        let manifest: RecoveryManifest = read_json(&directory.join(MANIFEST_FILE))?;
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
        let runtime = read_json(&directory.join(RUNTIME_FILE))?;
        Ok(Self {
            directory: directory.to_owned(),
            manifest,
            runtime,
        })
    }

    /// Immutable recovery manifest
    pub fn manifest(&self) -> &RecoveryManifest {
        &self.manifest
    }

    /// Validated wallet identity used by recovery backends
    pub fn verification_target(&self) -> Result<VerificationTarget, RecoverError> {
        VerificationTarget::new(
            self.manifest.target_fingerprint,
            self.manifest.master_xpub.clone(),
        )
    }

    /// Mutable runtime state
    pub fn runtime(&self) -> &RuntimeState {
        &self.runtime
    }

    /// Clone the durable next-candidate cursor
    pub fn cursor(&self) -> CandidateCursor {
        self.runtime.cursor.clone()
    }

    /// Whether unresolved matches prevent continued searching
    pub fn has_pending_matches(&self) -> bool {
        self.runtime
            .matches
            .iter()
            .any(|record| record.status == MatchStatus::Pending)
    }

    /// Whether a candidate identifier was already rejected
    pub fn is_rejected(&self, id: &CandidateId) -> bool {
        self.runtime
            .matches
            .iter()
            .any(|record| record.id == *id && record.status == MatchStatus::Rejected)
    }

    /// Persist a completed chunk, optional matches, and its next cursor atomically
    pub fn complete_chunk(
        &mut self,
        cursor: CandidateCursor,
        matches: Vec<MatchRecord>,
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

    /// Record a backend benchmark
    pub fn record_benchmark(&mut self, benchmark: BenchmarkRecord) -> Result<(), RecoverError> {
        self.runtime
            .benchmarks
            .retain(|record| record.backend != benchmark.backend);
        self.runtime.benchmarks.push(benchmark);
        self.save_runtime()
    }

    /// Mark one pending candidate as a false XFP collision
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

    /// Most recent benchmark for a backend
    pub fn latest_benchmark(&self, backend: BackendKind) -> Option<&BenchmarkRecord> {
        self.runtime
            .benchmarks
            .iter()
            .rev()
            .find(|record| record.backend == backend)
    }

    fn save_runtime(&self) -> Result<(), RecoverError> {
        atomic_write_json(&self.directory.join(RUNTIME_FILE), &self.runtime)
    }
}

fn validate_manifest(
    existing: &RecoveryManifest,
    expected: &RecoveryManifest,
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
    if existing.spec_hash != expected.spec_hash
        || existing.target_fingerprint != expected.target_fingerprint
        || existing.master_xpub != expected.master_xpub
        || existing.settings != expected.settings
        || existing.phases != expected.phases
    {
        return Err(RecoverError::StateMismatch);
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, RecoverError> {
    let mut file = File::open(path).map_err(|error| RecoverError::io(path, error))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| RecoverError::io(path, error))?;
    serde_json::from_slice(&bytes).map_err(|source| RecoverError::InvalidState {
        path: path.to_owned(),
        source,
    })
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), RecoverError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| RecoverError::InvalidState {
        path: path.to_owned(),
        source,
    })?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .map_err(|error| RecoverError::io(&temporary, error))?;
    file.write_all(&bytes)
        .map_err(|error| RecoverError::io(&temporary, error))?;
    file.write_all(b"\n")
        .map_err(|error| RecoverError::io(&temporary, error))?;
    file.sync_all()
        .map_err(|error| RecoverError::io(&temporary, error))?;
    fs::rename(&temporary, path).map_err(|error| RecoverError::io(path, error))?;
    sync_parent(path)
}

fn create_private_directory(path: &Path) -> Result<(), RecoverError> {
    fs::create_dir_all(path).map_err(|error| RecoverError::io(path, error))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| RecoverError::io(path, error))?;
    }
    Ok(())
}

fn protect_existing_file(path: &Path) -> Result<(), RecoverError> {
    if !path.exists() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|error| RecoverError::io(path, error))?;
    }
    Ok(())
}

fn check_private_state(directory: &Path) -> Result<(), RecoverError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        for path in [
            directory.to_owned(),
            directory.join(MANIFEST_FILE),
            directory.join(RUNTIME_FILE),
        ] {
            let metadata = fs::metadata(&path).map_err(|error| RecoverError::io(&path, error))?;
            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(RecoverError::InsecurePermissions(path));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = directory;
    Ok(())
}

fn sync_parent(path: &Path) -> Result<(), RecoverError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| RecoverError::io(parent, error))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    fn state(directory: &Path) -> RecoveryState {
        RecoveryState::open_or_create(
            directory,
            "spec".into(),
            "12345678".parse().unwrap(),
            RecoverySettings::default(),
            vec![PhaseSummary {
                phase: SearchPhase::WrittenLower,
                count: 10,
            }],
        )
        .unwrap()
    }

    fn benchmark(backend: BackendKind, checks_per_second: f64) -> BenchmarkRecord {
        BenchmarkRecord {
            backend,
            candidates: 1_024,
            batch_size: 65_536,
            workgroup_size: (backend != BackendKind::Cpu).then_some(64),
            cpu_share_percent: (backend == BackendKind::Hybrid).then_some(35),
            hardware_signature: Some("test-hardware".into()),
            seeds_per_second: checks_per_second,
            checks_per_second,
            verification_per_second: checks_per_second,
            generation_per_second: checks_per_second,
            device: "test-device".into(),
            measured_at_unix: 1,
        }
    }

    #[test]
    fn cursor_and_matches_commit_in_one_runtime_document() {
        let directory = tempdir().unwrap();
        let mut state = state(directory.path());
        let mut cursor = state.cursor();
        cursor.completed = 4;
        let candidate = Candidate::new(
            CandidateId("candidate".into()),
            SearchPhase::WrittenLower,
            vec!["alpha".into(), "brisk".into()],
        );
        state
            .complete_chunk(cursor, vec![MatchRecord::pending(&candidate)])
            .unwrap();

        let mut reopened = RecoveryState::open_existing(directory.path()).unwrap();
        assert_eq!(reopened.runtime().cursor.completed, 4);
        assert!(reopened.has_pending_matches());
        reopened.reject_match("candidate").unwrap();
        assert!(!reopened.has_pending_matches());
    }

    #[cfg(unix)]
    #[test]
    fn state_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let _state = state(directory.path());
        for name in [MANIFEST_FILE, RUNTIME_FILE] {
            let mode = fs::metadata(directory.path().join(name))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o077, 0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn opening_existing_state_rejects_loose_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let _state = state(directory.path());
        let runtime_path = directory.path().join(RUNTIME_FILE);
        fs::set_permissions(&runtime_path, fs::Permissions::from_mode(0o644)).unwrap();

        assert!(matches!(
            RecoveryState::open_existing(directory.path()),
            Err(RecoverError::InsecurePermissions(path)) if path == runtime_path
        ));
    }

    #[test]
    fn rejects_unversioned_state_without_migration() {
        let directory = tempdir().unwrap();
        let _state = state(directory.path());
        let manifest_path = directory.path().join(MANIFEST_FILE);
        let mut manifest = fs::read_to_string(&manifest_path).unwrap();
        manifest = manifest.replace("  \"state_format_version\": 2,\n", "");
        fs::write(&manifest_path, manifest).unwrap();

        assert!(matches!(
            RecoveryState::open_existing(directory.path()),
            Err(RecoverError::UnsupportedStateVersion(0))
        ));
    }

    #[test]
    fn benchmark_selection_is_unique_per_backend() {
        let directory = tempdir().unwrap();
        let mut state = state(directory.path());
        state
            .record_benchmark(benchmark(BackendKind::Cuda, 10.0))
            .unwrap();
        state
            .record_benchmark(benchmark(BackendKind::Cuda, 20.0))
            .unwrap();

        assert_eq!(state.runtime().benchmarks.len(), 1);
        assert_eq!(
            state
                .latest_benchmark(BackendKind::Cuda)
                .unwrap()
                .checks_per_second,
            20.0
        );
    }

    #[test]
    fn legacy_hybrid_configuration_uses_the_fixed_cpu_share() {
        let mut record = benchmark(BackendKind::Hybrid, 20.0);
        record.cpu_share_percent = None;

        assert_eq!(
            record
                .configuration()
                .unwrap()
                .cpu_share()
                .unwrap()
                .percent(),
            35
        );
    }
}
