use std::{borrow::Cow, fmt, path::Path, sync::OnceLock};

use bip39::{Language, Mnemonic};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::{
    domain::{CandidateId, VerificationTarget},
    error::RecoverError,
    input::{check_secret_file, read_secret},
};

const MISSING_WORD: u16 = u16::MAX;
const MNEMONIC_ALGORITHM_VERSION: u32 = 1;
const SUPPORTED_WORD_COUNTS: [usize; 5] = [12, 15, 18, 21, 24];

/// A normalized BIP39 passphrase whose debug representation is redacted
#[derive(Clone)]
pub struct SecretPassphrase(Zeroizing<String>);

impl SecretPassphrase {
    /// Construct and normalize a BIP39 passphrase
    pub fn new(value: String) -> Self {
        let mut normalized = Cow::Owned(value);
        Mnemonic::normalize_utf8_cow(&mut normalized);
        Self(Zeroizing::new(normalized.into_owned()))
    }

    /// Borrow the normalized passphrase
    pub fn expose(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for SecretPassphrase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretPassphrase([REDACTED])")
    }
}

/// Position-aware English BIP39 words with missing positions
#[derive(Clone)]
pub struct MnemonicTemplate {
    slots: Zeroizing<Vec<u16>>,
}

impl MnemonicTemplate {
    /// Parse one BIP39 word or `?` per line
    pub fn parse(text: &str) -> Result<Self, RecoverError> {
        let lines = text.lines().collect::<Vec<_>>();
        if !SUPPORTED_WORD_COUNTS.contains(&lines.len()) {
            return Err(RecoverError::InvalidSetting(format!(
                "mnemonic template must contain 12, 15, 18, 21, or 24 lines; found {}",
                lines.len()
            )));
        }

        let language = Language::English;
        let mut slots = Vec::with_capacity(lines.len());
        for (index, line) in lines.into_iter().enumerate() {
            let value = line.trim();
            if value == "?" {
                slots.push(MISSING_WORD);
                continue;
            }
            if value.is_empty() {
                return Err(RecoverError::InvalidSetting(format!(
                    "mnemonic template line {} is empty; use ? for a missing word",
                    index + 1
                )));
            }
            let normalized = value.to_ascii_lowercase();
            let word = language.find_word(&normalized).ok_or_else(|| {
                RecoverError::InvalidSetting(format!(
                    "mnemonic template line {} is not an English BIP39 word",
                    index + 1
                ))
            })?;
            slots.push(word);
        }

        Ok(Self {
            slots: Zeroizing::new(slots),
        })
    }

    /// Number of words in the complete mnemonic
    pub fn word_count(&self) -> usize {
        self.slots.len()
    }

    /// Number of known word positions
    pub fn known_word_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|word| **word != MISSING_WORD)
            .count()
    }

    /// Number of missing word positions
    pub fn missing_word_count(&self) -> usize {
        self.word_count() - self.known_word_count()
    }

    fn word_index(&self, position: usize) -> Option<u16> {
        let index = self.slots[position];
        (index != MISSING_WORD).then_some(index)
    }
}

impl fmt::Debug for MnemonicTemplate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MnemonicTemplate")
            .field("word_count", &self.word_count())
            .field("known_words", &self.known_word_count())
            .field("slots", &"[REDACTED]")
            .finish()
    }
}

/// Protected inputs for one mnemonic recovery
#[derive(Clone)]
pub struct MnemonicRecoveryInputs {
    /// Position-aware mnemonic template
    pub template: MnemonicTemplate,
    /// Known BIP39 passphrase, including an explicitly empty passphrase
    pub passphrase: SecretPassphrase,
}

impl fmt::Debug for MnemonicRecoveryInputs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MnemonicRecoveryInputs")
            .field("template", &self.template)
            .field("passphrase", &self.passphrase)
            .finish()
    }
}

/// Load an owner-only mnemonic template and explicit passphrase choice
pub fn load_mnemonic_recovery_inputs(
    template_path: &Path,
    passphrase_path: Option<&Path>,
    empty_passphrase: bool,
) -> Result<MnemonicRecoveryInputs, RecoverError> {
    check_secret_file(template_path)?;
    let template_text = Zeroizing::new(read_secret(template_path)?);
    let template = MnemonicTemplate::parse(&template_text)?;

    let passphrase = match (passphrase_path, empty_passphrase) {
        (Some(_), true) => {
            return Err(RecoverError::InvalidSetting(
                "passphrase-file conflicts with empty-passphrase".into(),
            ))
        }
        (None, false) => {
            return Err(RecoverError::InvalidSetting(
                "provide passphrase-file or explicitly select empty-passphrase".into(),
            ))
        }
        (None, true) => SecretPassphrase::new(String::new()),
        (Some(path), false) => {
            check_secret_file(path)?;
            let value = Zeroizing::new(read_secret(path)?);
            SecretPassphrase::new(value.trim_end_matches(['\r', '\n']).to_owned())
        }
    };

    Ok(MnemonicRecoveryInputs {
        template,
        passphrase,
    })
}

/// Big-integer cursor pointing to the next entropy assignment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MnemonicCursor {
    /// Rank of the next entropy assignment
    #[serde(with = "biguint_decimal")]
    pub next_rank: BigUint,
}

impl Default for MnemonicCursor {
    fn default() -> Self {
        Self {
            next_rank: BigUint::zero(),
        }
    }
}

/// One checksum-valid mnemonic candidate
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct MnemonicCandidate {
    #[zeroize(skip)]
    id: OnceLock<CandidateId>,
    #[zeroize(skip)]
    rank: BigUint,
    phrase: String,
}

impl MnemonicCandidate {
    /// Stable identifier for this exact mnemonic
    pub fn id(&self) -> &CandidateId {
        self.id.get_or_init(|| {
            let mut digest = Sha256::new();
            digest.update(b"recoverme-mnemonic-candidate\0");
            digest.update(MNEMONIC_ALGORITHM_VERSION.to_le_bytes());
            digest.update(self.phrase.as_bytes());
            CandidateId(hex::encode(digest.finalize()))
        })
    }

    /// Entropy-assignment rank that produced this mnemonic
    pub fn rank(&self) -> &BigUint {
        &self.rank
    }

    /// Borrow the normalized mnemonic phrase
    pub fn expose(&self) -> &str {
        &self.phrase
    }
}

impl fmt::Debug for MnemonicCandidate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MnemonicCandidate")
            .field("id", &self.id.get())
            .field("rank", &self.rank)
            .field("phrase", &"[REDACTED]")
            .finish()
    }
}

/// Deterministic checksum-aware search over missing BIP39 entropy bits
pub struct MnemonicPlan {
    template: MnemonicTemplate,
    base_entropy: Zeroizing<Vec<u8>>,
    unknown_entropy_positions: Vec<usize>,
    total_work: BigUint,
    checksum_bits: usize,
    checksum_is_known: bool,
}

impl MnemonicPlan {
    /// Compile a position-aware template into a deterministic entropy search
    pub fn compile(template: MnemonicTemplate) -> Result<Self, RecoverError> {
        let word_count = template.word_count();
        let entropy_bits = word_count / 3 * 32;
        let checksum_bits = word_count / 3;
        let mut base_entropy = Zeroizing::new(vec![0_u8; entropy_bits / 8]);
        let mut unknown_entropy_positions = Vec::new();

        for entropy_position in 0..entropy_bits {
            let word_position = entropy_position / 11;
            let bit_within_word = entropy_position % 11;
            let Some(word_index) = template.word_index(word_position) else {
                unknown_entropy_positions.push(entropy_position);
                continue;
            };
            let bit = (word_index >> (10 - bit_within_word)) & 1;
            if bit == 1 {
                set_entropy_bit(&mut base_entropy, entropy_position);
            }
        }

        let total_work = BigUint::one() << unknown_entropy_positions.len();
        let plan = Self {
            checksum_is_known: template.word_index(word_count - 1).is_some(),
            template,
            base_entropy,
            unknown_entropy_positions,
            total_work,
            checksum_bits,
        };
        if plan.unknown_entropy_positions.is_empty()
            && plan.candidate_at(&BigUint::zero())?.is_none()
        {
            return Err(RecoverError::InvalidSetting(
                "complete mnemonic template has an invalid BIP39 checksum".into(),
            ));
        }
        Ok(plan)
    }

    /// Original mnemonic template
    pub fn template(&self) -> &MnemonicTemplate {
        &self.template
    }

    /// Number of unknown entropy bits enumerated by this plan
    pub fn unknown_entropy_bits(&self) -> usize {
        self.unknown_entropy_positions.len()
    }

    /// Number of checksum bits in the configured BIP39 length
    pub fn checksum_bits(&self) -> usize {
        self.checksum_bits
    }

    /// Whether the final known word constrains checksum bits
    pub fn checksum_is_known(&self) -> bool {
        self.checksum_is_known
    }

    /// Exact number of entropy assignments visited by the cursor
    pub fn total_work(&self) -> &BigUint {
        &self.total_work
    }

    /// Expected number of checksum-valid mnemonics
    pub fn expected_candidates(&self) -> BigUint {
        if self.unknown_entropy_positions.is_empty() {
            return BigUint::one();
        }
        if self.checksum_is_known {
            let expected = &self.total_work >> self.checksum_bits;
            expected.max(BigUint::one())
        } else {
            self.total_work.clone()
        }
    }

    /// Generate up to `limit` valid candidates and advance over rejected checksums
    pub fn next_batch(
        &self,
        cursor: &mut MnemonicCursor,
        limit: usize,
    ) -> Result<Vec<MnemonicCandidate>, RecoverError> {
        let mut candidates = Vec::with_capacity(limit);
        while candidates.len() < limit && cursor.next_rank < self.total_work {
            let rank = cursor.next_rank.clone();
            cursor.next_rank += 1_u8;
            if let Some(candidate) = self.candidate_at(&rank)? {
                candidates.push(candidate);
            }
        }
        Ok(candidates)
    }

    /// Reconstruct a checksum-valid candidate from its entropy-assignment rank
    pub fn candidate_at(&self, rank: &BigUint) -> Result<Option<MnemonicCandidate>, RecoverError> {
        if rank >= &self.total_work {
            return Err(RecoverError::InvalidSetting(
                "mnemonic candidate rank is outside the configured search".into(),
            ));
        }
        let mut entropy = self.base_entropy.clone();
        let unknown_count = self.unknown_entropy_positions.len();
        for (index, position) in self.unknown_entropy_positions.iter().copied().enumerate() {
            let rank_bit = unknown_count - index - 1;
            if rank.bit(rank_bit as u64) {
                set_entropy_bit(&mut entropy, position);
            }
        }

        let mnemonic = Mnemonic::from_entropy_in(Language::English, &entropy)
            .map_err(|error| RecoverError::InvalidMnemonic(error.to_string()))?;
        let words = Language::English.word_list();
        for (position, actual) in mnemonic.word_iter().enumerate() {
            if self
                .template
                .word_index(position)
                .is_some_and(|expected| words[expected as usize] != actual)
            {
                return Ok(None);
            }
        }

        Ok(Some(MnemonicCandidate {
            id: OnceLock::new(),
            rank: rank.clone(),
            phrase: mnemonic.to_string(),
        }))
    }
}

/// Compute a stable identifier for mnemonic recovery secrets and target
pub fn mnemonic_spec_hash(inputs: &MnemonicRecoveryInputs, target: &VerificationTarget) -> String {
    let mut digest = Sha256::new();
    digest.update(b"recoverme-mnemonic-spec-v1\0");
    digest.update((inputs.template.word_count() as u64).to_le_bytes());
    for slot in inputs.template.slots.iter() {
        digest.update(slot.to_le_bytes());
    }
    digest.update(inputs.passphrase.expose().as_bytes());
    digest.update([0]);
    digest.update(target.fingerprint().bytes());
    if let Some(master_xpub) = target.master_xpub() {
        digest.update(master_xpub.public_key());
        digest.update(master_xpub.chain_code());
    }
    hex::encode(digest.finalize())
}

fn set_entropy_bit(entropy: &mut [u8], position: usize) {
    entropy[position / 8] |= 1 << (7 - position % 8);
}

mod biguint_decimal {
    use num_bigint::BigUint;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &BigUint, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_str_radix(10))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        BigUint::parse_bytes(value.as_bytes(), 10)
            .ok_or_else(|| serde::de::Error::custom("invalid decimal big integer"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_TWELVE: &str = "abandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabout";

    #[test]
    fn accepts_every_standard_word_count() {
        for word_count in SUPPORTED_WORD_COUNTS {
            let template = std::iter::repeat_n("?", word_count)
                .collect::<Vec<_>>()
                .join("\n");
            assert_eq!(
                MnemonicTemplate::parse(&template).unwrap().word_count(),
                word_count
            );
        }
    }

    #[test]
    fn rejects_nonstandard_word_counts_and_non_bip39_words() {
        assert!(MnemonicTemplate::parse(
            &std::iter::repeat_n("?", 11).collect::<Vec<_>>().join("\n")
        )
        .is_err());
        assert!(MnemonicTemplate::parse(&VALID_TWELVE.replace("about", "notaword")).is_err());
    }

    #[test]
    fn complete_template_produces_its_single_mnemonic() {
        let plan = MnemonicPlan::compile(MnemonicTemplate::parse(VALID_TWELVE).unwrap()).unwrap();
        let candidate = plan.candidate_at(&BigUint::zero()).unwrap().unwrap();

        assert_eq!(plan.unknown_entropy_bits(), 0);
        assert_eq!(plan.total_work(), &BigUint::one());
        assert_eq!(candidate.expose(), VALID_TWELVE.replace('\n', " "));
    }

    #[test]
    fn missing_final_word_in_twelve_words_has_seven_unknown_bits() {
        let template = VALID_TWELVE.replace("about", "?");
        let plan = MnemonicPlan::compile(MnemonicTemplate::parse(&template).unwrap()).unwrap();
        let mut cursor = MnemonicCursor::default();
        let candidates = plan.next_batch(&mut cursor, 128).unwrap();

        assert_eq!(plan.unknown_entropy_bits(), 7);
        assert_eq!(plan.total_work(), &BigUint::from(128_u16));
        assert_eq!(candidates.len(), 128);
        assert_eq!(cursor.next_rank, BigUint::from(128_u16));
    }

    #[test]
    fn known_final_word_filters_invalid_checksums() {
        let template = VALID_TWELVE.replacen("abandon", "?", 1);
        let plan = MnemonicPlan::compile(MnemonicTemplate::parse(&template).unwrap()).unwrap();
        let mut cursor = MnemonicCursor::default();
        let candidates = plan.next_batch(&mut cursor, 2_048).unwrap();

        assert_eq!(plan.unknown_entropy_bits(), 11);
        assert!((64..=192).contains(&candidates.len()));
        assert_eq!(cursor.next_rank, BigUint::from(2_048_u16));
        assert!(candidates
            .iter()
            .any(|candidate| candidate.expose() == VALID_TWELVE.replace('\n', " ")));
    }

    #[test]
    fn eight_known_words_and_missing_final_word_is_a_forty_bit_search() {
        let template = format!(
            "{}\n?\n?\n?\n?",
            VALID_TWELVE.lines().take(8).collect::<Vec<_>>().join("\n")
        );
        let plan = MnemonicPlan::compile(MnemonicTemplate::parse(&template).unwrap()).unwrap();

        assert_eq!(plan.template().known_word_count(), 8);
        assert_eq!(plan.unknown_entropy_bits(), 40);
        assert_eq!(plan.total_work(), &(BigUint::one() << 40));
    }

    #[test]
    fn every_known_word_count_changes_the_twelve_word_entropy_space() {
        let valid_words = VALID_TWELVE.lines().collect::<Vec<_>>();
        for known_words in 0..=12 {
            let template = valid_words
                .iter()
                .enumerate()
                .map(
                    |(position, word)| {
                        if position < known_words {
                            *word
                        } else {
                            "?"
                        }
                    },
                )
                .collect::<Vec<_>>()
                .join("\n");
            let plan = MnemonicPlan::compile(MnemonicTemplate::parse(&template).unwrap()).unwrap();
            let expected_unknown_bits = 128_usize.saturating_sub((known_words * 11).min(128));

            assert_eq!(plan.template().known_word_count(), known_words);
            assert_eq!(plan.unknown_entropy_bits(), expected_unknown_bits);
        }
    }

    #[test]
    fn final_word_entropy_width_tracks_each_standard_checksum_width() {
        for word_count in SUPPORTED_WORD_COUNTS {
            let entropy_bytes = word_count / 3 * 4;
            let mnemonic = Mnemonic::from_entropy(&vec![0_u8; entropy_bytes]).unwrap();
            let mut words = mnemonic.word_iter().collect::<Vec<_>>();
            words[word_count - 1] = "?";
            let plan =
                MnemonicPlan::compile(MnemonicTemplate::parse(&words.join("\n")).unwrap()).unwrap();

            assert_eq!(plan.checksum_bits(), word_count / 3);
            assert_eq!(plan.unknown_entropy_bits(), 11 - word_count / 3);
            assert_eq!(
                plan.total_work(),
                &(BigUint::one() << (11 - word_count / 3))
            );
        }
    }
}
