use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use num_bigint::BigUint;
use pbkdf2::pbkdf2_hmac;
use rayon::prelude::*;
use sha2::Sha512;
use zeroize::Zeroizing;

use crate::{
    crypto::{matching_candidate_indices_for_target, SeedBatch},
    domain::{BackendKind, VerificationTarget},
    error::RecoverError,
    mnemonic::{
        MnemonicCandidate, MnemonicCursor, MnemonicPlan, MnemonicRecoveryInputs, SecretPassphrase,
    },
    mnemonic_state::{MnemonicMatchRecord, MnemonicRecoveryState},
    state::BenchmarkRecord,
};

/// Terminal reason returned by a mnemonic recovery run
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnemonicRunOutcome {
    /// Every entropy assignment was exhausted
    Exhausted,
    /// The user requested a clean stop between batches
    Interrupted,
    /// Strong wallet-identity matches were recorded
    MatchesFound(usize),
}

struct MnemonicCpuBackend {
    salt: Zeroizing<Vec<u8>>,
    batch_size: usize,
}

impl MnemonicCpuBackend {
    fn new(passphrase: &SecretPassphrase, batch_size: usize) -> Self {
        let mut salt = Zeroizing::new(Vec::with_capacity(8 + passphrase.expose().len()));
        salt.extend_from_slice(b"mnemonic");
        salt.extend_from_slice(passphrase.expose().as_bytes());
        Self {
            salt,
            batch_size: batch_size.max(1),
        }
    }

    fn derive_seeds(&self, candidates: &[MnemonicCandidate]) -> SeedBatch {
        let seeds = candidates
            .par_iter()
            .map(|candidate| {
                let mut seed = [0_u8; 64];
                pbkdf2_hmac::<Sha512>(candidate.expose().as_bytes(), &self.salt, 2_048, &mut seed);
                seed
            })
            .collect();
        SeedBatch::new(seeds)
    }

    fn verify(
        &self,
        candidates: &[MnemonicCandidate],
        target: &VerificationTarget,
    ) -> Result<Vec<usize>, RecoverError> {
        let seeds = self.derive_seeds(candidates);
        matching_candidate_indices_for_target(&seeds, target)
    }
}

/// Benchmark CPU mnemonic generation, seed derivation, and master-XPUB verification
pub fn benchmark_mnemonic_backend(
    plan: &MnemonicPlan,
    state: &mut MnemonicRecoveryState,
    inputs: &MnemonicRecoveryInputs,
    sample_size: usize,
) -> Result<BenchmarkRecord, RecoverError> {
    let backend = MnemonicCpuBackend::new(
        &inputs.passphrase,
        rayon::current_num_threads().max(1) * 128,
    );
    let mut cursor = MnemonicCursor::default();
    let generation_started = Instant::now();
    let candidates = plan.next_batch(&mut cursor, sample_size.max(1))?;
    let generation_elapsed = generation_started.elapsed();
    if candidates.is_empty() {
        return Err(RecoverError::InvalidSetting(
            "mnemonic plan contains no checksum-valid candidates".into(),
        ));
    }

    let warmup_count = candidates.len().min(4);
    let _warmup = backend.derive_seeds(&candidates[..warmup_count]);
    let seed_started = Instant::now();
    let _seeds = backend.derive_seeds(&candidates);
    let seed_elapsed = seed_started.elapsed();
    let target = state.verification_target();
    let verify_started = Instant::now();
    let _matches = backend.verify(&candidates, &target)?;
    let verify_elapsed = verify_started.elapsed();
    let count = candidates.len();
    let sustained_elapsed = generation_elapsed.max(verify_elapsed);
    let record = BenchmarkRecord {
        backend: BackendKind::Cpu,
        candidates: count,
        batch_size: backend.batch_size,
        workgroup_size: None,
        cpu_share_percent: None,
        hardware_signature: Some(format!(
            "CPU; rayon-threads={}",
            rayon::current_num_threads()
        )),
        seeds_per_second: count as f64 / seed_elapsed.as_secs_f64(),
        checks_per_second: count as f64 / sustained_elapsed.as_secs_f64(),
        verification_per_second: count as f64 / verify_elapsed.as_secs_f64(),
        generation_per_second: count as f64 / generation_elapsed.as_secs_f64(),
        device: format!("CPU ({} Rayon threads)", rayon::current_num_threads()),
        measured_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    state.record_benchmark(record.clone())?;
    Ok(record)
}

/// Execute or resume a CPU mnemonic recovery
pub fn run_mnemonic_recovery(
    plan: &MnemonicPlan,
    state: &mut MnemonicRecoveryState,
    inputs: &MnemonicRecoveryInputs,
    stop: Arc<AtomicBool>,
) -> Result<MnemonicRunOutcome, RecoverError> {
    if state.has_pending_matches() {
        return Err(RecoverError::PendingMatches);
    }
    let batch_size = state.latest_benchmark().map_or_else(
        || rayon::current_num_threads().max(1) * 128,
        |record| record.batch_size,
    );
    let backend = MnemonicCpuBackend::new(&inputs.passphrase, batch_size);
    let target = state.verification_target();
    let mut prepared = prepare_batch(plan, state.cursor(), backend.batch_size)?;

    loop {
        if prepared.candidates.is_empty() {
            state.complete_chunk(prepared.next_cursor, Vec::new())?;
            return Ok(MnemonicRunOutcome::Exhausted);
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(MnemonicRunOutcome::Interrupted);
        }

        let next_start = prepared.next_cursor.clone();
        let (indices, next) = std::thread::scope(|scope| {
            let producer = scope.spawn(|| prepare_batch(plan, next_start, backend.batch_size));
            let indices = backend.verify(&prepared.candidates, &target)?;
            let next = producer
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)??;
            Ok::<_, RecoverError>((indices, next))
        })?;
        let matches = indices
            .into_iter()
            .map(|index| MnemonicMatchRecord::pending(&prepared.candidates[index]))
            .collect::<Vec<_>>();
        let match_count = matches.len();
        state.complete_chunk(prepared.next_cursor, matches)?;
        log::info!(
            "Checked mnemonic entropy assignments completed={}",
            state.cursor().next_rank
        );
        if match_count > 0 {
            return Ok(MnemonicRunOutcome::MatchesFound(match_count));
        }
        prepared = next;
    }
}

struct PreparedMnemonicBatch {
    candidates: Vec<MnemonicCandidate>,
    next_cursor: MnemonicCursor,
}

fn prepare_batch(
    plan: &MnemonicPlan,
    mut cursor: MnemonicCursor,
    batch_size: usize,
) -> Result<PreparedMnemonicBatch, RecoverError> {
    let candidates = plan.next_batch(&mut cursor, batch_size)?;
    Ok(PreparedMnemonicBatch {
        candidates,
        next_cursor: cursor,
    })
}

/// Remaining entropy assignments from a durable cursor
pub fn remaining_work(plan: &MnemonicPlan, cursor: &MnemonicCursor) -> BigUint {
    if cursor.next_rank >= *plan.total_work() {
        BigUint::default()
    } else {
        plan.total_work() - &cursor.next_rank
    }
}

#[cfg(test)]
mod tests {
    use bip32::{Prefix, XPrv};
    use bip39::{Language, Mnemonic};
    use tempfile::tempdir;

    use super::*;
    use crate::{
        domain::MasterXpubTarget,
        mnemonic::{mnemonic_spec_hash, MnemonicTemplate},
    };

    #[test]
    fn recovers_a_missing_final_word_and_checkpoints_the_match() {
        let phrase =
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let mnemonic = Mnemonic::parse_in(Language::English, phrase).unwrap();
        let xpub = XPrv::new(mnemonic.to_seed("secret")).unwrap().public_key();
        let target = VerificationTarget::from_master_xpub(
            MasterXpubTarget::parse(&xpub.to_string(Prefix::XPUB)).unwrap(),
        );
        let template = MnemonicTemplate::parse(
            "abandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\nabandon\n?",
        )
        .unwrap();
        let inputs = MnemonicRecoveryInputs {
            template: template.clone(),
            passphrase: SecretPassphrase::new("secret".into()),
        };
        let plan = MnemonicPlan::compile(template).unwrap();
        let directory = tempdir().unwrap();
        let mut state = MnemonicRecoveryState::open_or_create(
            directory.path(),
            mnemonic_spec_hash(&inputs, &target),
            target,
            &plan,
        )
        .unwrap();

        let outcome =
            run_mnemonic_recovery(&plan, &mut state, &inputs, Arc::new(AtomicBool::new(false)))
                .unwrap();

        assert_eq!(outcome, MnemonicRunOutcome::MatchesFound(1));
        let match_record = &state.runtime().matches[0];
        let recovered = plan
            .candidate_at(&match_record.parsed_rank().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(recovered.expose(), phrase);
    }
}
