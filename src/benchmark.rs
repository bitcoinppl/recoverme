use std::{
    num::{NonZeroU32, NonZeroUsize},
    time::Instant,
};

use crate::{
    backend::create_deriver,
    domain::{
        BackendKind, CandidateBatch, CandidateCursor, SearchPhase, SecretMnemonic,
        VerificationTarget,
    },
    error::RecoverError,
    search::RecoveryPlan,
};

/// Wallet identity comparison performed by a benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationMode {
    /// Compare the four-byte master fingerprint
    Fingerprint,
    /// Filter by chain code and confirm the complete master public key
    MasterXpub,
}

impl VerificationMode {
    /// Identify the comparison mode represented by a verification target
    pub const fn from_target(target: &VerificationTarget) -> Self {
        match target {
            VerificationTarget::Fingerprint(_) => Self::Fingerprint,
            VerificationTarget::MasterXpub { .. } => Self::MasterXpub,
        }
    }
}

impl std::fmt::Display for VerificationMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Fingerprint => "xfp",
            Self::MasterXpub => "master-xpub",
        })
    }
}

/// Configuration for a sustained end-to-end backend measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SustainedBenchmarkConfig {
    /// Candidates checked in each timed iteration
    pub sample_size: NonZeroUsize,
    /// Number of timed pipeline iterations
    pub repetitions: NonZeroUsize,
    /// Candidate batch size configured on the backend
    pub batch_size: NonZeroUsize,
    /// Accelerator workgroup size when applicable
    pub workgroup_size: Option<NonZeroU32>,
    /// Last recovery phase included in candidate generation
    pub through: SearchPhase,
}

/// Sustained end-to-end throughput from one backend and target mode
#[derive(Debug, Clone)]
pub struct SustainedBenchmarkResult {
    /// Backend measured
    pub backend: BackendKind,
    /// Wallet identity comparison measured
    pub verification_mode: VerificationMode,
    /// Device/runtime description
    pub device: String,
    /// Candidates checked in each timed iteration
    pub sample_size: usize,
    /// Number of timed pipeline iterations
    pub repetitions: usize,
    /// Candidate batch size configured on the backend
    pub batch_size: usize,
    /// Accelerator workgroup size when applicable
    pub workgroup_size: Option<u32>,
    /// Aggregate checks per second with candidate preparation overlapped
    pub sustained_checks_per_second: f64,
    /// Median checks per second across timed iterations
    pub median_checks_per_second: f64,
    /// Lowest checks per second across timed iterations
    pub minimum_checks_per_second: f64,
    /// Highest checks per second across timed iterations
    pub maximum_checks_per_second: f64,
    /// Total matching candidate indices returned across timed iterations
    pub matches: usize,
    /// Per-iteration wall-clock durations in milliseconds
    pub iteration_milliseconds: Vec<f64>,
}

/// Measure repeated production-style candidate generation and verification
pub fn benchmark_sustained_pipeline(
    plan: &RecoveryPlan,
    mnemonic: &SecretMnemonic,
    target: &VerificationTarget,
    backend: BackendKind,
    config: SustainedBenchmarkConfig,
) -> Result<SustainedBenchmarkResult, RecoverError> {
    let sample_size = config.sample_size.get();
    let repetitions = config.repetitions.get();
    let measured_batches =
        repetitions
            .checked_add(1)
            .ok_or(RecoverError::InsufficientBenchmarkCandidates {
                required: usize::MAX,
                available: plan.count_through(config.through)?,
            })?;
    let required = sample_size.checked_mul(measured_batches).ok_or(
        RecoverError::InsufficientBenchmarkCandidates {
            required: usize::MAX,
            available: plan.count_through(config.through)?,
        },
    )?;
    let available = plan.count_through(config.through)?;
    if available < required as u128 {
        return Err(RecoverError::InsufficientBenchmarkCandidates {
            required,
            available,
        });
    }

    let mut deriver = create_deriver(backend, mnemonic)?;
    deriver.configure(
        config.batch_size.get(),
        config.workgroup_size.map(NonZeroU32::get),
    )?;

    let mut cursor = CandidateCursor::default();
    let mut prepared = next_exact_batch(plan, &mut cursor, config.through, sample_size)?;

    // a complete warmup excludes lazy runtime setup and first kernel compilation
    let _warmup_matches = deriver.verify(&prepared, target)?;

    let aggregate_started = Instant::now();
    let mut iteration_milliseconds = Vec::with_capacity(repetitions);
    let mut matches = 0;
    for _ in 0..repetitions {
        let iteration_started = Instant::now();
        let next_cursor = cursor.clone();
        let (indices, next, advanced_cursor) = std::thread::scope(|scope| {
            let producer = scope.spawn(move || {
                let mut advanced_cursor = next_cursor;
                let batch =
                    next_exact_batch(plan, &mut advanced_cursor, config.through, sample_size)?;
                Ok::<_, RecoverError>((batch, advanced_cursor))
            });
            let indices = deriver.verify(&prepared, target)?;
            let (next, advanced_cursor) = producer
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)??;
            Ok::<_, RecoverError>((indices, next, advanced_cursor))
        })?;
        matches += indices.len();
        prepared = next;
        cursor = advanced_cursor;
        iteration_milliseconds.push(iteration_started.elapsed().as_secs_f64() * 1_000.0);
    }
    let aggregate_seconds = aggregate_started.elapsed().as_secs_f64();
    let mut rates = iteration_milliseconds
        .iter()
        .map(|milliseconds| sample_size as f64 * 1_000.0 / milliseconds)
        .collect::<Vec<_>>();
    rates.sort_by(f64::total_cmp);

    Ok(SustainedBenchmarkResult {
        backend,
        verification_mode: VerificationMode::from_target(target),
        device: deriver.device_name(),
        sample_size,
        repetitions,
        batch_size: deriver.preferred_batch_size(),
        workgroup_size: deriver.workgroup_size(),
        sustained_checks_per_second: sample_size as f64 * repetitions as f64 / aggregate_seconds,
        median_checks_per_second: median(&rates),
        minimum_checks_per_second: rates[0],
        maximum_checks_per_second: rates[rates.len() - 1],
        matches,
        iteration_milliseconds,
    })
}

fn next_exact_batch(
    plan: &RecoveryPlan,
    cursor: &mut CandidateCursor,
    through: SearchPhase,
    sample_size: usize,
) -> Result<CandidateBatch, RecoverError> {
    let candidates = plan.next_batch(cursor, through, sample_size)?;
    if candidates.len() != sample_size {
        return Err(RecoverError::InsufficientBenchmarkCandidates {
            required: sample_size,
            available: candidates.len() as u128,
        });
    }
    CandidateBatch::new(candidates)
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};

    use super::*;

    #[test]
    fn median_uses_the_center_of_odd_and_even_samples() {
        assert_eq!(median(&[1.0, 3.0, 8.0]), 3.0);
        assert_eq!(median(&[1.0, 3.0, 8.0, 10.0]), 5.5);
    }

    #[test]
    fn rejects_a_plan_too_small_for_sustained_measurement() {
        let words = crate::WrittenWords::new(vec!["alpha".into()]).unwrap();
        let settings = crate::RecoverySettings {
            max_replacements: 0,
            ..crate::RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();
        let mnemonic = SecretMnemonic::new(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
                .into(),
        );
        let target = VerificationTarget::Fingerprint("00000000".parse().unwrap());
        let config = SustainedBenchmarkConfig {
            sample_size: NonZeroUsize::new(4).unwrap(),
            repetitions: NonZeroUsize::new(2).unwrap(),
            batch_size: NonZeroUsize::new(4).unwrap(),
            workgroup_size: NonZeroU32::new(32),
            through: SearchPhase::WrittenCase,
        };

        assert!(matches!(
            benchmark_sustained_pipeline(&plan, &mnemonic, &target, BackendKind::Cpu, config),
            Err(RecoverError::InsufficientBenchmarkCandidates { .. })
        ));
    }
}
