use bip32::XPrv;
use bip39::{Language, Mnemonic};
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;
use rayon::prelude::*;
use sha2::Sha512;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::{
    domain::{BackendKind, CandidateBatch, SecretMnemonic, TargetFingerprint, VerificationTarget},
    error::RecoverError,
};

/// Secret BIP39 seeds corresponding positionally to a candidate batch
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SeedBatch(Vec<[u8; 64]>);

impl SeedBatch {
    /// Construct a seed batch after backend derivation
    pub fn new(seeds: Vec<[u8; 64]>) -> Self {
        Self(seeds)
    }

    /// Borrow the derived seeds
    pub fn as_slice(&self) -> &[[u8; 64]] {
        &self.0
    }
}

/// Backend capable of deriving and verifying candidate passphrases
pub trait RecoveryBackend {
    /// Backend identity recorded in benchmarks and state
    fn kind(&self) -> BackendKind;

    /// Human-readable runtime or device description
    fn device_name(&self) -> String;

    /// Batch size suitable for this backend
    fn preferred_batch_size(&self) -> usize;

    /// Workgroup size used by an accelerated backend
    fn workgroup_size(&self) -> Option<u32> {
        None
    }

    /// Apply a benchmark-selected batch and optional workgroup size
    fn configure(
        &mut self,
        batch_size: usize,
        workgroup_size: Option<u32>,
    ) -> Result<(), RecoverError> {
        if batch_size == 0 {
            return Err(RecoverError::InvalidSetting(
                "backend batch size must be greater than zero".into(),
            ));
        }
        if workgroup_size.is_some() {
            return Err(RecoverError::InvalidSetting(
                "CPU backend does not accept a workgroup size".into(),
            ));
        }
        Ok(())
    }

    /// Derive one seed for every candidate in the input batch
    fn derive_seeds(&mut self, candidates: &CandidateBatch) -> Result<SeedBatch, RecoverError>;

    /// Return indices that match the configured wallet identity
    fn verify(
        &mut self,
        candidates: &CandidateBatch,
        target: &VerificationTarget,
    ) -> Result<Vec<usize>, RecoverError> {
        let seeds = self.derive_seeds(candidates)?;
        matching_candidate_indices_for_target(&seeds, target)
    }
}

/// Audited CPU seed derivation using the `bip39` crate and Rayon
pub struct CpuSeedDeriver {
    mnemonic: Zeroizing<Vec<u8>>,
    batch_size: usize,
}

impl CpuSeedDeriver {
    /// Parse a validated secret mnemonic for repeated derivations
    pub fn new(mnemonic: &SecretMnemonic) -> Result<Self, RecoverError> {
        let mnemonic = Mnemonic::parse_in(Language::English, mnemonic.expose())
            .map_err(|error| RecoverError::InvalidMnemonic(error.to_string()))?;
        Ok(Self {
            mnemonic: Zeroizing::new(mnemonic.to_string().into_bytes()),
            batch_size: rayon::current_num_threads().max(1) * 128,
        })
    }
}

impl RecoveryBackend for CpuSeedDeriver {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn device_name(&self) -> String {
        format!("CPU ({} Rayon threads)", rayon::current_num_threads())
    }

    fn preferred_batch_size(&self) -> usize {
        self.batch_size
    }

    fn configure(
        &mut self,
        batch_size: usize,
        workgroup_size: Option<u32>,
    ) -> Result<(), RecoverError> {
        if batch_size == 0 {
            return Err(RecoverError::InvalidSetting(
                "backend batch size must be greater than zero".into(),
            ));
        }
        if workgroup_size.is_some() {
            return Err(RecoverError::InvalidSetting(
                "CPU backend does not accept a workgroup size".into(),
            ));
        }
        self.batch_size = batch_size;
        Ok(())
    }

    fn derive_seeds(&mut self, candidates: &CandidateBatch) -> Result<SeedBatch, RecoverError> {
        let seeds = candidates
            .candidates()
            .par_iter()
            .map(|candidate| {
                let passphrase = candidate.passphrase().as_bytes();
                let mut salt = Zeroizing::new([0_u8; 108]);
                salt[..8].copy_from_slice(b"mnemonic");
                salt[8..8 + passphrase.len()].copy_from_slice(passphrase);
                let mut seed = [0_u8; 64];
                pbkdf2_hmac::<Sha512>(
                    &self.mnemonic,
                    &salt[..8 + passphrase.len()],
                    2_048,
                    &mut seed,
                );
                seed
            })
            .collect();
        Ok(SeedBatch::new(seeds))
    }
}

/// Return candidate indices whose BIP32 master fingerprint matches the target
pub fn matching_candidate_indices(
    seeds: &SeedBatch,
    target: TargetFingerprint,
) -> Result<Vec<usize>, RecoverError> {
    seeds
        .as_slice()
        .par_iter()
        .enumerate()
        .map(|(index, seed)| {
            fingerprint_from_seed(seed)
                .map(|fingerprint| (fingerprint == target.bytes()).then_some(index))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|matches| matches.into_iter().flatten().collect())
}

/// Return candidate indices matching the configured wallet identity
pub fn matching_candidate_indices_for_target(
    seeds: &SeedBatch,
    target: &VerificationTarget,
) -> Result<Vec<usize>, RecoverError> {
    match target {
        VerificationTarget::Fingerprint(fingerprint) => {
            matching_candidate_indices(seeds, *fingerprint)
        }
        VerificationTarget::MasterXpub { master_xpub, .. } => {
            let expected_chain_code = master_xpub.chain_code();
            let expected_public_key = master_xpub.public_key();
            let master_hmac = Hmac::<Sha512>::new_from_slice(b"Bitcoin seed")
                .expect("BIP32 master HMAC accepts its fixed key");
            seeds
                .as_slice()
                .par_iter()
                .enumerate()
                .map_init(
                    || master_hmac.clone(),
                    |hmac_template, (index, seed)| {
                        let mut hmac = hmac_template.clone();
                        hmac.update(seed);
                        let digest = Zeroizing::new(hmac.finalize().into_bytes().to_vec());
                        if digest[32..] != expected_chain_code {
                            return Ok(None);
                        }
                        let private_key = XPrv::new(seed).map_err(|error| {
                            RecoverError::FingerprintDerivation(error.to_string())
                        })?;
                        Ok((private_key.public_key().to_bytes() == expected_public_key)
                            .then_some(index))
                    },
                )
                .collect::<Result<Vec<_>, RecoverError>>()
                .map(|matches| matches.into_iter().flatten().collect())
        }
    }
}

/// Derive the standard four-byte BIP32 master public-key fingerprint
pub fn fingerprint_from_seed(seed: &[u8; 64]) -> Result<[u8; 4], RecoverError> {
    let private_key =
        XPrv::new(seed).map_err(|error| RecoverError::FingerprintDerivation(error.to_string()))?;
    Ok(private_key.public_key().fingerprint())
}

/// Derive a fingerprint directly from a mnemonic and passphrase
pub fn fingerprint_for_passphrase(
    mnemonic: &SecretMnemonic,
    passphrase: &str,
) -> Result<[u8; 4], RecoverError> {
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic.expose())
        .map_err(|error| RecoverError::InvalidMnemonic(error.to_string()))?;
    let seed = mnemonic.to_seed(passphrase);
    fingerprint_from_seed(&seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Candidate, MasterXpubTarget};
    use bip32::Prefix;

    const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

    #[test]
    fn preserves_coldcard_fingerprint_byte_order() {
        let mnemonic = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());

        assert_eq!(
            hex::encode(fingerprint_for_passphrase(&mnemonic, "").unwrap()),
            "5436d724"
        );
        assert_eq!(
            hex::encode(
                fingerprint_for_passphrase(&mnemonic, "alphabriskcactusdaringeagerfabricgadget",)
                    .unwrap()
            ),
            "997f3522"
        );
    }

    #[test]
    fn accelerated_cpu_seed_derivation_matches_bip39() {
        let secret = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let mut deriver = CpuSeedDeriver::new(&secret).unwrap();
        let candidates = [
            "",
            "lowercase",
            "TitleCase",
            "UPPERCASE",
            "mIxEdCapitalization",
            &"a".repeat(100),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, passphrase)| {
            Candidate::new(
                crate::domain::CandidateId(index.to_string()),
                crate::domain::SearchPhase::WrittenCase,
                vec![passphrase.to_owned()],
            )
        })
        .collect::<Vec<_>>();
        let batch = CandidateBatch::new(candidates).unwrap();
        let actual = deriver.derive_seeds(&batch).unwrap();
        let reference = Mnemonic::parse_in(Language::English, PUBLIC_TEST_MNEMONIC).unwrap();
        for (candidate, seed) in batch.candidates().iter().zip(actual.as_slice()) {
            assert_eq!(*seed, reference.to_seed(candidate.passphrase()));
        }
    }

    #[test]
    fn master_xpub_target_rejects_fingerprint_only_collisions() {
        let mnemonic = Mnemonic::parse_in(Language::English, PUBLIC_TEST_MNEMONIC).unwrap();
        let expected = XPrv::new(mnemonic.to_seed("correct")).unwrap();
        let master_xpub =
            MasterXpubTarget::parse(&expected.public_key().to_string(Prefix::XPUB)).unwrap();
        let fingerprint = hex::encode(master_xpub.fingerprint())
            .parse::<TargetFingerprint>()
            .unwrap();
        let target = VerificationTarget::new(fingerprint, Some(master_xpub)).unwrap();
        let seeds = SeedBatch::new(vec![mnemonic.to_seed("wrong"), mnemonic.to_seed("correct")]);

        assert_eq!(
            matching_candidate_indices_for_target(&seeds, &target).unwrap(),
            [1]
        );
    }
}
