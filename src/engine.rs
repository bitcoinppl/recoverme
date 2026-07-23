use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use crate::{
    backend::create_deriver,
    domain::{
        BackendKind, CandidateBatch, CandidateCursor, CandidateId, SearchPhase, SecretMnemonic,
        VerificationTarget,
    },
    error::RecoverError,
    search::RecoveryPlan,
    state::{BenchmarkRecord, MatchRecord, RecoveryState},
};

/// Terminal reason returned by a recovery run
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    /// Every requested phase was exhausted without a pending match
    Exhausted,
    /// The user requested a clean stop between chunks
    Interrupted,
    /// One or more wallet candidates were recorded for manual verification
    MatchesFound(usize),
}

/// Benchmark seed derivation and complete fingerprint verification
pub fn benchmark_backend(
    plan: &RecoveryPlan,
    state: &mut RecoveryState,
    mnemonic: &SecretMnemonic,
    backend: BackendKind,
    sample_size: usize,
) -> Result<BenchmarkRecord, RecoverError> {
    benchmark_backend_with_config(plan, state, mnemonic, backend, sample_size, None, None)
}

/// Benchmark a backend with an explicit runtime configuration
pub fn benchmark_backend_with_config(
    plan: &RecoveryPlan,
    state: &mut RecoveryState,
    mnemonic: &SecretMnemonic,
    backend: BackendKind,
    sample_size: usize,
    batch_size: Option<usize>,
    workgroup_size: Option<u32>,
) -> Result<BenchmarkRecord, RecoverError> {
    let mut deriver = create_deriver(backend, mnemonic)?;
    if let Some(batch_size) = batch_size {
        deriver.configure(batch_size, workgroup_size)?;
    } else if workgroup_size.is_some() {
        deriver.configure(deriver.preferred_batch_size(), workgroup_size)?;
    }
    let sample_size = sample_size.max(1);
    let mut cursor = CandidateCursor::default();
    let generation_started = Instant::now();
    let candidates = plan.next_batch(&mut cursor, SearchPhase::WrittenCase, sample_size)?;
    if candidates.is_empty() {
        return Err(RecoverError::SeedDerivation(
            "recovery plan produced no benchmark candidates".into(),
        ));
    }

    let candidates = CandidateBatch::new(candidates)?;
    let generation_elapsed = generation_started.elapsed();
    let warmup_count = candidates.len().min(deriver.preferred_batch_size().min(4));
    let warmup = CandidateBatch::new(candidates.candidates()[..warmup_count].to_vec())?;
    let _warmup = deriver.derive_seeds(&warmup)?;
    let seed_started = Instant::now();
    let _seeds = deriver.derive_seeds(&candidates)?;
    let seed_elapsed = seed_started.elapsed();
    let verify_started = Instant::now();
    let target = state.verification_target()?;
    let _matches = deriver.verify(&candidates, &target)?;
    let verify_elapsed = verify_started.elapsed();
    let pipeline_started = Instant::now();
    let _next_candidates = std::thread::scope(|scope| {
        let producer = scope.spawn(|| {
            let mut next_cursor = cursor;
            let next =
                plan.next_batch(&mut next_cursor, SearchPhase::WrittenCase, candidates.len())?;
            CandidateBatch::new(next)
        });
        deriver.verify(&candidates, &target)?;
        producer
            .join()
            .map_err(|_| RecoverError::CandidatePreparationPanic)?
    })?;
    let pipeline_elapsed = pipeline_started.elapsed();

    let benchmark = BenchmarkRecord {
        backend,
        candidates: candidates.len(),
        batch_size: deriver.preferred_batch_size(),
        workgroup_size: deriver.workgroup_size(),
        seeds_per_second: candidates.len() as f64 / seed_elapsed.as_secs_f64(),
        checks_per_second: candidates.len() as f64 / pipeline_elapsed.as_secs_f64(),
        verification_per_second: candidates.len() as f64 / verify_elapsed.as_secs_f64(),
        generation_per_second: candidates.len() as f64 / generation_elapsed.as_secs_f64(),
        device: deriver.device_name(),
        measured_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    state.record_benchmark(benchmark.clone())?;
    Ok(benchmark)
}

/// Execute or resume deterministic chunks through the requested phase
pub fn run_recovery(
    plan: &RecoveryPlan,
    state: &mut RecoveryState,
    mnemonic: &SecretMnemonic,
    target: &VerificationTarget,
    backend: BackendKind,
    through: SearchPhase,
    stop: Arc<AtomicBool>,
) -> Result<RunOutcome, RecoverError> {
    if state.has_pending_matches() {
        return Err(RecoverError::PendingMatches);
    }

    let mut deriver = create_deriver(backend, mnemonic)?;
    if let Some(benchmark) = state.latest_benchmark(backend) {
        deriver.configure(benchmark.batch_size, benchmark.workgroup_size)?;
    }
    let batch_size = deriver.preferred_batch_size().max(1);
    let rejected = state
        .runtime()
        .matches
        .iter()
        .filter(|record| record.status == crate::state::MatchStatus::Rejected)
        .map(|record| record.id.clone())
        .collect::<HashSet<_>>();
    let mut prepared = prepare_batch(plan, state.cursor(), through, batch_size, &rejected)?;

    loop {
        if prepared.exhausted {
            state.complete_chunk(prepared.next_cursor, Vec::new())?;
            return Ok(RunOutcome::Exhausted);
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(RunOutcome::Interrupted);
        }
        let next_start = prepared.next_cursor.clone();
        let (indices, next) = std::thread::scope(|scope| {
            let producer =
                scope.spawn(|| prepare_batch(plan, next_start, through, batch_size, &rejected));
            let indices = if prepared.batch.is_empty() {
                Ok(Vec::new())
            } else {
                deriver.verify(&prepared.batch, target)
            };
            let next = producer
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)?;
            Ok::<_, RecoverError>((indices?, next?))
        })?;
        let matches = indices
            .into_iter()
            .map(|index| MatchRecord::pending(&prepared.batch.candidates()[index]))
            .collect::<Vec<_>>();
        let match_count = matches.len();
        state.complete_chunk(prepared.next_cursor.clone(), matches)?;
        log::info!(
            "Checked candidates completed={} backend={backend}",
            prepared.next_cursor.completed
        );

        if match_count > 0 {
            return Ok(RunOutcome::MatchesFound(match_count));
        }
        prepared = next;
    }
}

struct PreparedBatch {
    batch: CandidateBatch,
    next_cursor: CandidateCursor,
    exhausted: bool,
}

fn prepare_batch(
    plan: &RecoveryPlan,
    mut cursor: CandidateCursor,
    through: SearchPhase,
    batch_size: usize,
    rejected: &HashSet<CandidateId>,
) -> Result<PreparedBatch, RecoverError> {
    let candidates = plan.next_batch(&mut cursor, through, batch_size)?;
    let exhausted = candidates.is_empty();
    let candidates = if rejected.is_empty() {
        candidates
    } else {
        candidates
            .into_iter()
            .filter(|candidate| !rejected.contains(candidate.id()))
            .collect()
    };
    Ok(PreparedBatch {
        batch: CandidateBatch::new(candidates)?,
        next_cursor: cursor,
        exhausted,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use tempfile::tempdir;

    use super::*;
    use crate::{
        crypto::fingerprint_for_passphrase,
        domain::{RecoverySettings, TargetFingerprint, WrittenWords},
        input::{recovery_spec_hash, RecoveryInputs},
    };

    const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

    #[test]
    fn match_is_checkpointed_and_can_be_rejected_before_resume() {
        let directory = tempdir().unwrap();
        let mnemonic = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let words = WrittenWords::new(vec!["alpha".into()]).unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings.clone()).unwrap();
        let fingerprint = hex::encode(fingerprint_for_passphrase(&mnemonic, "alpha").unwrap())
            .parse::<TargetFingerprint>()
            .unwrap();
        let inputs = RecoveryInputs {
            mnemonic: mnemonic.clone(),
            written_words: words,
        };
        let mut state = RecoveryState::open_or_create(
            directory.path(),
            recovery_spec_hash(&inputs, fingerprint, &settings),
            fingerprint,
            settings,
            plan.phase_summaries(),
        )
        .unwrap();

        let outcome = run_recovery(
            &plan,
            &mut state,
            &mnemonic,
            &VerificationTarget::Fingerprint(fingerprint),
            BackendKind::Cpu,
            SearchPhase::WrittenLower,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        assert_eq!(outcome, RunOutcome::MatchesFound(1));
        let record = &state.runtime().matches[0];
        assert_eq!(record.passphrase, "alpha");
        assert_eq!(record.words, ["alpha"]);

        let match_id = record.id.0.clone();
        state.reject_match(&match_id).unwrap();
        let resumed = run_recovery(
            &plan,
            &mut state,
            &mnemonic,
            &VerificationTarget::Fingerprint(fingerprint),
            BackendKind::Cpu,
            SearchPhase::WrittenLower,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        assert_eq!(resumed, RunOutcome::Exhausted);

        let reopened = RecoveryState::open_existing(directory.path()).unwrap();
        assert_eq!(reopened.runtime().cursor.completed, 1);
        assert!(!reopened.has_pending_matches());
    }

    #[test]
    fn interruption_does_not_checkpoint_prepared_candidates() {
        let directory = tempdir().unwrap();
        let mnemonic = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings.clone()).unwrap();
        let fingerprint = "00000000".parse::<TargetFingerprint>().unwrap();
        let inputs = RecoveryInputs {
            mnemonic: mnemonic.clone(),
            written_words: words,
        };
        let mut state = RecoveryState::open_or_create(
            directory.path(),
            recovery_spec_hash(&inputs, fingerprint, &settings),
            fingerprint,
            settings,
            plan.phase_summaries(),
        )
        .unwrap();
        let stop = Arc::new(AtomicBool::new(true));

        let outcome = run_recovery(
            &plan,
            &mut state,
            &mnemonic,
            &VerificationTarget::Fingerprint(fingerprint),
            BackendKind::Cpu,
            SearchPhase::WrittenLower,
            stop,
        )
        .unwrap();

        assert_eq!(outcome, RunOutcome::Interrupted);
        assert_eq!(state.runtime().cursor, CandidateCursor::default());
    }
}
