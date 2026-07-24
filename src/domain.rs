use std::{collections::HashSet, fmt, num::NonZeroUsize, str::FromStr, sync::OnceLock};

use bip32::XPub;
use clap::ValueEnum;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::error::RecoverError;

/// Version of the candidate ordering and checkpoint format
pub const ALGORITHM_VERSION: u32 = 3;

/// Maximum passphrase length accepted by Coldcard
pub const DEFAULT_MAX_PASSPHRASE_BYTES: usize = 100;

/// Settings that define the immutable candidate space
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoverySettings {
    /// Number of nearest BIP39 words retained for each written token
    pub neighbors_per_word: usize,
    /// Maximum number of written tokens that may be replaced
    pub max_replacements: usize,
    /// Adjacent-swap distance emitted before exhaustive lexical permutations
    pub local_swap_radius: usize,
    /// Maximum candidate passphrase length in bytes
    pub max_passphrase_bytes: usize,
    /// Whether lowercase-only phases were completed by an earlier search
    pub lowercase_already_tried: bool,
    /// Candidate token ordering strategy
    pub order: OrderMode,
    /// Spaces inserted before candidate tokens
    pub spacing: SpacingMode,
    /// Whether the all-concatenated spacing pattern was already exhausted
    pub concatenated_already_tried: bool,
}

impl Default for RecoverySettings {
    fn default() -> Self {
        Self {
            neighbors_per_word: 3,
            max_replacements: 2,
            local_swap_radius: 3,
            max_passphrase_bytes: DEFAULT_MAX_PASSPHRASE_BYTES,
            lowercase_already_tried: false,
            order: OrderMode::Permuted,
            spacing: SpacingMode::Concatenated,
            concatenated_already_tried: false,
        }
    }
}

/// Candidate token ordering strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum OrderMode {
    /// Keep surviving recipe slots in their written order
    Written,
    /// Rank nearby swaps first, then exhaust every unique permutation
    Permuted,
}

/// Candidate spacing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SpacingMode {
    /// Concatenate every token
    Concatenated,
    /// Insert one space between every token
    Between,
    /// Try concatenated and conventionally spaced candidates
    Both,
    /// Try every Coldcard Add Word leading-space combination
    Coldcard,
}

impl fmt::Display for OrderMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Written => "written",
            Self::Permuted => "permuted",
        })
    }
}

impl fmt::Display for SpacingMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Concatenated => "concatenated",
            Self::Between => "between",
            Self::Both => "both",
            Self::Coldcard => "coldcard",
        })
    }
}

/// Ordered recovery phases, from most likely to largest search space
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum SearchPhase {
    /// Written words, lowercase, in every unique order
    WrittenLower,
    /// Written words with lowercase, Title, and UPPER variants
    WrittenCase,
    /// A fixed number of nearest-word substitutions, lowercase
    NeighborLower(usize),
    /// A fixed number of nearest-word substitutions with capitalization variants
    NeighborCase(usize),
}

impl SearchPhase {
    /// Number of substitutions represented by this phase
    pub const fn replacement_count(self) -> usize {
        match self {
            Self::WrittenLower | Self::WrittenCase => 0,
            Self::NeighborLower(count) | Self::NeighborCase(count) => count,
        }
    }

    /// Whether this phase enumerates capitalization variants
    pub const fn includes_case_variants(self) -> bool {
        matches!(self, Self::WrittenCase | Self::NeighborCase(_))
    }

    /// Position of this phase in a plan with the given substitution limit
    pub const fn index(self, max_replacements: usize) -> usize {
        match self {
            Self::WrittenLower => 0,
            Self::WrittenCase => 1,
            Self::NeighborLower(count) => 1 + count,
            Self::NeighborCase(count) => 1 + max_replacements + count,
        }
    }

    /// Next phase in a plan with the given substitution limit
    pub const fn next(self, max_replacements: usize) -> Option<Self> {
        match self {
            Self::WrittenLower => Some(Self::WrittenCase),
            Self::WrittenCase if max_replacements > 0 => Some(Self::NeighborLower(1)),
            Self::WrittenCase => None,
            Self::NeighborLower(count) if count < max_replacements => {
                Some(Self::NeighborLower(count + 1))
            }
            Self::NeighborLower(_) => Some(Self::NeighborCase(1)),
            Self::NeighborCase(count) if count < max_replacements => {
                Some(Self::NeighborCase(count + 1))
            }
            Self::NeighborCase(_) => None,
        }
    }

    /// All phases for a substitution limit in execution order
    pub fn all(max_replacements: usize) -> Vec<Self> {
        let mut phases = vec![Self::WrittenLower, Self::WrittenCase];
        phases.extend((1..=max_replacements).map(Self::NeighborLower));
        phases.extend((1..=max_replacements).map(Self::NeighborCase));
        phases
    }
}

impl fmt::Display for SearchPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::WrittenLower => "written-lower",
            Self::WrittenCase => "written-case",
            Self::NeighborLower(count) => return write!(formatter, "neighbor-{count}-lower"),
            Self::NeighborCase(count) => return write!(formatter, "neighbor-{count}-case"),
        };
        formatter.write_str(value)
    }
}

impl FromStr for SearchPhase {
    type Err = RecoverError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "written-lower" => return Ok(Self::WrittenLower),
            "written-case" => return Ok(Self::WrittenCase),
            _ => {}
        }
        let (count, variants) = value
            .strip_prefix("neighbor-")
            .and_then(|suffix| {
                suffix
                    .strip_suffix("-lower")
                    .map(|count| (count, false))
                    .or_else(|| suffix.strip_suffix("-case").map(|count| (count, true)))
            })
            .ok_or_else(|| RecoverError::InvalidSetting("invalid search phase".into()))?;
        let count = count
            .parse::<usize>()
            .map_err(|_| RecoverError::InvalidSetting("invalid search phase".into()))?;
        if count == 0 {
            return Err(RecoverError::InvalidSetting(
                "neighbor phase count must be positive".into(),
            ));
        }
        Ok(if variants {
            Self::NeighborCase(count)
        } else {
            Self::NeighborLower(count)
        })
    }
}

impl From<SearchPhase> for String {
    fn from(phase: SearchPhase) -> Self {
        phase.to_string()
    }
}

impl TryFrom<String> for SearchPhase {
    type Error = RecoverError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Runtime used to derive BIP39 seeds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    /// Select the fastest verified backend available in this build
    Auto,
    /// Audited Rust BIP39 implementation, parallelized with Rayon
    Cpu,
    /// CubeCL's CPU runtime using the same kernel as GPU backends
    CubeCpu,
    /// CubeCL's Metal runtime
    Metal,
    /// Concurrent CPU and fastest available Metal runtime
    Hybrid,
    /// CubeCL's CUDA runtime
    Cuda,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::CubeCpu => "cube-cpu",
            Self::Metal => "metal",
            Self::Hybrid => "hybrid",
            Self::Cuda => "cuda",
        };
        formatter.write_str(value)
    }
}

/// Nonzero candidate count processed as one backend batch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchSize(NonZeroUsize);

impl BatchSize {
    /// Return the validated batch size
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for BatchSize {
    type Error = RecoverError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value).map(Self).ok_or_else(|| {
            RecoverError::InvalidSetting("backend batch size must be greater than zero".into())
        })
    }
}

/// CubeCL workgroup size supported by the recovery kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize(u32);

impl WorkgroupSize {
    /// Return the validated workgroup size
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for WorkgroupSize {
    type Error = RecoverError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if !(32..=256).contains(&value) || !value.is_power_of_two() {
            return Err(RecoverError::InvalidSetting(
                "CubeCL workgroup size must be a power of two from 32 through 256".into(),
            ));
        }
        Ok(Self(value))
    }
}

/// Nonzero CPU percentage assigned by an explicit hybrid backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuShare(u8);

impl CpuShare {
    /// Return the validated CPU percentage
    pub const fn percent(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for CpuShare {
    type Error = RecoverError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if !(1..=99).contains(&value) {
            return Err(RecoverError::InvalidSetting(
                "hybrid CPU share must be from 1 through 99 percent".into(),
            ));
        }
        Ok(Self(value))
    }
}

/// Validated runtime configuration accepted by recovery backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendConfiguration {
    /// CPU-only configuration
    Cpu {
        /// Candidate batch size
        batch_size: BatchSize,
    },
    /// CubeCL accelerator configuration without CPU participation
    Cube {
        /// Candidate batch size
        batch_size: BatchSize,
        /// Accelerator workgroup size
        workgroup_size: WorkgroupSize,
    },
    /// Concurrent CPU and CubeCL accelerator configuration
    Hybrid {
        /// Candidate batch size
        batch_size: BatchSize,
        /// Accelerator workgroup size
        workgroup_size: WorkgroupSize,
        /// Percentage of candidates assigned to the CPU
        cpu_share: CpuShare,
    },
}

impl BackendConfiguration {
    /// Construct a CPU-only configuration
    pub fn cpu(batch_size: usize) -> Result<Self, RecoverError> {
        Ok(Self::Cpu {
            batch_size: batch_size.try_into()?,
        })
    }

    /// Construct a CubeCL accelerator configuration
    pub fn cube(batch_size: usize, workgroup_size: u32) -> Result<Self, RecoverError> {
        Ok(Self::Cube {
            batch_size: batch_size.try_into()?,
            workgroup_size: workgroup_size.try_into()?,
        })
    }

    /// Construct a concurrent CPU and CubeCL accelerator configuration
    pub fn hybrid(
        batch_size: usize,
        workgroup_size: u32,
        cpu_share_percent: u8,
    ) -> Result<Self, RecoverError> {
        Ok(Self::Hybrid {
            batch_size: batch_size.try_into()?,
            workgroup_size: workgroup_size.try_into()?,
            cpu_share: cpu_share_percent.try_into()?,
        })
    }

    /// Candidate batch size shared by every configuration
    pub const fn batch_size(self) -> BatchSize {
        match self {
            Self::Cpu { batch_size }
            | Self::Cube { batch_size, .. }
            | Self::Hybrid { batch_size, .. } => batch_size,
        }
    }

    /// Optional accelerator workgroup size
    pub const fn workgroup_size(self) -> Option<WorkgroupSize> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cube { workgroup_size, .. } | Self::Hybrid { workgroup_size, .. } => {
                Some(workgroup_size)
            }
        }
    }

    /// Optional hybrid CPU share
    pub const fn cpu_share(self) -> Option<CpuShare> {
        match self {
            Self::Cpu { .. } | Self::Cube { .. } => None,
            Self::Hybrid { cpu_share, .. } => Some(cpu_share),
        }
    }
}

/// Validated depth-zero master extended public key
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct MasterXpubTarget(String);

impl MasterXpubTarget {
    /// Parse and validate a serialized master public key
    pub fn parse(value: &str) -> Result<Self, RecoverError> {
        let value = value.trim();
        let xpub = value
            .parse::<XPub>()
            .map_err(|error| RecoverError::InvalidMasterXpub(error.to_string()))?;
        if xpub.attrs().depth != 0
            || xpub.attrs().parent_fingerprint != [0; 4]
            || xpub.attrs().child_number.0 != 0
        {
            return Err(RecoverError::InvalidMasterXpub(
                "extended public key must be the depth-zero master key".into(),
            ));
        }
        Ok(Self(value.to_owned()))
    }

    /// Master public-key fingerprint in Coldcard display order
    pub fn fingerprint(&self) -> [u8; 4] {
        self.xpub().fingerprint()
    }

    /// Master BIP32 chain code
    pub fn chain_code(&self) -> [u8; 32] {
        self.xpub().attrs().chain_code
    }

    /// Compressed master secp256k1 public key
    pub fn public_key(&self) -> [u8; 33] {
        self.xpub().to_bytes()
    }

    fn xpub(&self) -> XPub {
        self.0
            .parse()
            .expect("MasterXpubTarget is validated at construction")
    }
}

impl<'de> Deserialize<'de> for MasterXpubTarget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

/// Wallet identity used to verify derived BIP39 seeds
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationTarget {
    /// Compare the four-byte master fingerprint on the CPU
    Fingerprint(TargetFingerprint),
    /// Filter by master chain code and confirm the complete master public key
    MasterXpub {
        /// Expected fingerprint, retained for Coldcard display and validation
        fingerprint: TargetFingerprint,
        /// Validated master public key
        master_xpub: MasterXpubTarget,
    },
}

impl VerificationTarget {
    /// Construct a target and reject a mismatched master public key
    pub fn new(
        fingerprint: TargetFingerprint,
        master_xpub: Option<MasterXpubTarget>,
    ) -> Result<Self, RecoverError> {
        let Some(master_xpub) = master_xpub else {
            return Ok(Self::Fingerprint(fingerprint));
        };
        if master_xpub.fingerprint() != fingerprint.bytes() {
            return Err(RecoverError::MasterXpubFingerprintMismatch);
        }
        Ok(Self::MasterXpub {
            fingerprint,
            master_xpub,
        })
    }

    /// Four-byte fingerprint shown by Coldcard
    pub const fn fingerprint(&self) -> TargetFingerprint {
        match self {
            Self::Fingerprint(fingerprint) | Self::MasterXpub { fingerprint, .. } => *fingerprint,
        }
    }

    /// Optional master public key for chain-code filtering
    pub fn master_xpub(&self) -> Option<&MasterXpubTarget> {
        match self {
            Self::Fingerprint(_) => None,
            Self::MasterXpub { master_xpub, .. } => Some(master_xpub),
        }
    }
}

/// Four-byte BIP32 master public-key fingerprint shown by Coldcard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetFingerprint([u8; 4]);

impl TargetFingerprint {
    /// Return the fingerprint bytes in Coldcard display order
    pub const fn bytes(self) -> [u8; 4] {
        self.0
    }
}

impl FromStr for TargetFingerprint {
    type Err = RecoverError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(RecoverError::InvalidFingerprint(value.to_owned()));
        }

        let decoded =
            hex::decode(value).map_err(|_| RecoverError::InvalidFingerprint(value.into()))?;
        let bytes = decoded
            .try_into()
            .map_err(|_| RecoverError::InvalidFingerprint(value.into()))?;
        Ok(Self(bytes))
    }
}

impl fmt::Display for TargetFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&hex::encode(self.0))
    }
}

/// Validated secret mnemonic whose debug representation is always redacted
#[derive(Clone)]
pub struct SecretMnemonic(Zeroizing<String>);

impl SecretMnemonic {
    /// Construct a secret mnemonic from validated text
    pub fn new(value: String) -> Self {
        Self(Zeroizing::new(value))
    }

    /// Borrow the mnemonic text
    pub fn expose(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for SecretMnemonic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretMnemonic([REDACTED])")
    }
}

/// Written passphrase tokens in their original positional order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenWords(Vec<String>);

impl WrittenWords {
    /// Construct a nonempty ordered collection of normalized words
    pub fn new(words: Vec<String>) -> Result<Self, RecoverError> {
        if words.is_empty() {
            return Err(RecoverError::NoWrittenWords);
        }
        Ok(Self(words))
    }

    /// Borrow the normalized words
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Number of written words
    pub fn word_count(&self) -> usize {
        self.0.len()
    }
}

/// One position in an advanced recovery recipe
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSlot {
    alternatives: Vec<String>,
    optional: bool,
}

impl TokenSlot {
    /// Construct a slot from ranked normalized alternatives
    pub fn new(alternatives: Vec<String>, optional: bool) -> Result<Self, RecoverError> {
        if alternatives.is_empty() {
            return Err(RecoverError::InvalidSetting(
                "recipe slots require at least one alternative".into(),
            ));
        }
        if alternatives
            .iter()
            .any(|value| value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_alphabetic()))
        {
            return Err(RecoverError::InvalidSetting(
                "recipe alternatives must contain ASCII letters only".into(),
            ));
        }
        let normalized = alternatives
            .into_iter()
            .map(|value| value.to_ascii_lowercase())
            .collect::<Vec<_>>();
        let mut seen = HashSet::new();
        if normalized.iter().any(|value| !seen.insert(value.clone())) {
            return Err(RecoverError::InvalidSetting(
                "recipe alternatives must be unique within a slot".into(),
            ));
        }
        Ok(Self {
            alternatives: normalized,
            optional,
        })
    }

    /// Ranked alternatives, with the written value first
    pub fn alternatives(&self) -> &[String] {
        &self.alternatives
    }

    /// Whether this slot may be omitted
    pub const fn is_optional(&self) -> bool {
        self.optional
    }
}

/// Typed passphrase recovery recipe
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryRecipe(Vec<TokenSlot>);

impl RecoveryRecipe {
    /// Construct a nonempty recipe
    pub fn new(slots: Vec<TokenSlot>) -> Result<Self, RecoverError> {
        if slots.is_empty() {
            return Err(RecoverError::NoWrittenWords);
        }
        Ok(Self(slots))
    }

    /// Construct required singleton slots from a simple words file
    pub fn from_written_words(words: &WrittenWords) -> Self {
        Self(
            words
                .as_slice()
                .iter()
                .map(|word| TokenSlot {
                    alternatives: vec![word.clone()],
                    optional: false,
                })
                .collect(),
        )
    }

    /// Ordered recipe slots
    pub fn slots(&self) -> &[TokenSlot] {
        &self.0
    }
}

/// Stable identifier for exact passphrase bytes supplied to BIP39
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CandidateId(pub String);

impl fmt::Display for CandidateId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Candidate passphrase and its readable word segmentation
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct Candidate {
    #[zeroize(skip)]
    id: OnceLock<CandidateId>,
    #[zeroize(skip)]
    phase: SearchPhase,
    passphrase: String,
    words: Vec<String>,
}

/// Reusable packed representation of an ordered candidate batch
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct CandidateBatch {
    candidates: Vec<Candidate>,
    bytes: Vec<u8>,
    #[zeroize(skip)]
    lengths: Vec<u16>,
    #[zeroize(skip)]
    stride: usize,
}

impl CandidateBatch {
    /// Pack candidates into fixed-width bytes for CPU and accelerator backends
    pub fn new(candidates: Vec<Candidate>) -> Result<Self, RecoverError> {
        let stride = candidates
            .iter()
            .map(|candidate| candidate.passphrase().len())
            .max()
            .unwrap_or(0)
            .max(1);
        if stride > DEFAULT_MAX_PASSPHRASE_BYTES {
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.passphrase().len() > DEFAULT_MAX_PASSPHRASE_BYTES)
                .expect("the maximum passphrase belongs to a candidate");
            return Err(RecoverError::SeedDerivation(format!(
                "candidate {} exceeds {DEFAULT_MAX_PASSPHRASE_BYTES} bytes",
                candidate.id()
            )));
        }
        let mut bytes = vec![0_u8; candidates.len() * stride];
        let mut lengths = Vec::with_capacity(candidates.len());
        for (index, candidate) in candidates.iter().enumerate() {
            let passphrase = candidate.passphrase().as_bytes();
            let start = index * stride;
            bytes[start..start + passphrase.len()].copy_from_slice(passphrase);
            lengths.push(passphrase.len() as u16);
        }
        Ok(Self {
            candidates,
            bytes,
            lengths,
            stride,
        })
    }

    /// Candidate metadata in the same order as packed bytes
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Contiguous fixed-stride candidate bytes
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Actual byte length of every candidate
    pub fn lengths(&self) -> &[u16] {
        &self.lengths
    }

    /// Byte width reserved for each candidate in the packed buffer
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Number of candidates in the batch
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Whether the batch contains no candidates
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

impl Candidate {
    /// Construct a candidate from already transformed words
    pub fn new(id: CandidateId, phase: SearchPhase, words: Vec<String>) -> Self {
        let passphrase = words.concat();
        Self {
            id: OnceLock::from(id),
            phase,
            passphrase,
            words,
        }
    }

    /// Stable candidate identifier
    pub fn id(&self) -> &CandidateId {
        self.id
            .get_or_init(|| CandidateId::for_passphrase(self.passphrase.as_bytes()))
    }

    /// Search phase that produced this candidate
    pub const fn phase(&self) -> SearchPhase {
        self.phase
    }

    /// Exact no-space passphrase tested by BIP39
    pub fn passphrase(&self) -> &str {
        &self.passphrase
    }

    /// Readable segmentation of the passphrase
    pub fn words(&self) -> &[String] {
        &self.words
    }

    pub(crate) fn from_passphrase(
        phase: SearchPhase,
        passphrase: String,
        words: Vec<String>,
    ) -> Self {
        Self {
            id: OnceLock::new(),
            phase,
            passphrase,
            words,
        }
    }
}

impl CandidateId {
    fn for_passphrase(passphrase: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"recoverme-candidate\0");
        digest.update(ALGORITHM_VERSION.to_le_bytes());
        digest.update(passphrase);
        Self(hex::encode(digest.finalize()))
    }
}

/// Cursor within a phase's permutation stream
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum PermutationCursor {
    /// Ranked local adjacent-swap prefix
    Local { index: usize },
    /// Exhaustive lexicographic multiset permutation rank
    Lexical {
        #[serde(with = "u128_string")]
        rank: u128,
    },
}

impl Default for PermutationCursor {
    fn default() -> Self {
        Self::Local { index: 0 }
    }
}

/// Serializable cursor pointing at the next candidate to test
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateCursor {
    /// Current phase
    pub phase: SearchPhase,
    /// Replacement-base index within the phase
    pub base_index: usize,
    /// Permutation cursor within the current base
    pub permutation: PermutationCursor,
    /// Case-pattern rank within the current permutation
    #[serde(with = "u128_string")]
    pub case_rank: u128,
    /// Spacing-pattern rank for the current transformed permutation
    #[serde(with = "u128_string")]
    pub spacing_rank: u128,
    /// Number of verified candidates across all phases
    #[serde(with = "u128_string")]
    pub completed: u128,
}

impl Default for CandidateCursor {
    fn default() -> Self {
        Self {
            phase: SearchPhase::WrittenLower,
            base_index: 0,
            permutation: PermutationCursor::default(),
            case_rank: 0,
            spacing_rank: 0,
            completed: 0,
        }
    }
}

/// Exact candidate count for one search phase
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseSummary {
    /// Search phase
    pub phase: SearchPhase,
    /// Unique candidates represented by this phase
    #[serde(with = "u128_string")]
    pub count: u128,
}

pub(crate) mod u128_string {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_configuration_rejects_impossible_runtime_settings() {
        assert!(BackendConfiguration::cpu(0).is_err());
        assert!(BackendConfiguration::cube(1, 31).is_err());
        assert!(BackendConfiguration::cube(1, 48).is_err());
        assert!(BackendConfiguration::cube(1, 512).is_err());
        assert!(BackendConfiguration::hybrid(1, 64, 0).is_err());
        assert!(BackendConfiguration::hybrid(1, 64, 100).is_err());

        let configuration = BackendConfiguration::hybrid(65_536, 128, 5).unwrap();
        assert_eq!(configuration.batch_size().get(), 65_536);
        assert_eq!(configuration.workgroup_size().unwrap().get(), 128);
        assert_eq!(configuration.cpu_share().unwrap().percent(), 5);
    }
}
