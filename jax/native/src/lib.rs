//! Typed Python bridge to recoverme's Rust recovery core

use std::{path::Path, str::FromStr};

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{create_exception, exceptions::PyException, prelude::*};
use recoverme::{
    crypto::{matching_candidate_indices, SeedBatch},
    domain::{
        Candidate, CandidateCursor, RecoverySettings, SearchPhase, TargetFingerprint,
        DEFAULT_MAX_PASSPHRASE_BYTES,
    },
    input::{load_inputs, recovery_spec_hash, RecoveryInputs},
    search::RecoveryPlan,
    state::{MatchRecord, MatchStatus, RecoveryState},
    RecoverError,
};
use zeroize::{Zeroize, Zeroizing};

create_exception!(_native, RecoveryError, PyException);

type CandidateRecord = (String, String, String, Vec<String>);

struct PendingBatch {
    token: u64,
    cursor: CandidateCursor,
    candidates: Vec<Candidate>,
}

/// Fixed-width candidate buffers for one opaque recovery batch
#[pyclass(frozen)]
struct PreparedBatch {
    token: u64,
    count: usize,
    candidate_bytes: Py<PyArray2<u8>>,
    lengths: Py<PyArray1<u16>>,
}

#[pymethods]
impl PreparedBatch {
    #[getter]
    const fn token(&self) -> u64 {
        self.token
    }

    #[getter]
    const fn count(&self) -> usize {
        self.count
    }

    #[getter]
    fn candidate_bytes(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.candidate_bytes.clone_ref(py)
    }

    #[getter]
    fn lengths(&self, py: Python<'_>) -> Py<PyArray1<u16>> {
        self.lengths.clone_ref(py)
    }
}

/// Result of fingerprint verification and atomic checkpoint completion
#[pyclass(frozen)]
struct BatchCompletion {
    checked: usize,
    matches: usize,
    completed: u128,
}

#[pymethods]
impl BatchCompletion {
    #[getter]
    const fn checked(&self) -> usize {
        self.checked
    }

    #[getter]
    const fn matches(&self) -> usize {
        self.matches
    }

    #[getter]
    const fn completed(&self) -> u128 {
        self.completed
    }
}

/// Shared recovery planner and durable state owner
#[pyclass(unsendable)]
struct RecoverySession {
    plan: RecoveryPlan,
    state: RecoveryState,
    mnemonic: Zeroizing<Vec<u8>>,
    mnemonic_taken: bool,
    pending: Option<PendingBatch>,
    next_token: u64,
}

#[pymethods]
impl RecoverySession {
    /// Create a new shared Rust/JAX recovery state or reopen a matching one
    #[staticmethod]
    #[pyo3(signature = (
        mnemonic_file,
        words_file,
        fingerprint,
        state_dir,
        neighbors=3,
        max_replacements=2,
        lowercase_already_tried=false
    ))]
    fn plan(
        mnemonic_file: &str,
        words_file: &str,
        fingerprint: &str,
        state_dir: &str,
        neighbors: usize,
        max_replacements: usize,
        lowercase_already_tried: bool,
    ) -> PyResult<Self> {
        let fingerprint = TargetFingerprint::from_str(fingerprint).map_err(to_python_error)?;
        let inputs = load_inputs(Path::new(mnemonic_file), Path::new(words_file))
            .map_err(to_python_error)?;
        let settings = RecoverySettings {
            neighbors_per_word: neighbors,
            max_replacements,
            lowercase_already_tried,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&inputs.written_words, settings.clone())
            .map_err(to_python_error)?;
        let state = RecoveryState::open_or_create(
            Path::new(state_dir),
            recovery_spec_hash(&inputs, fingerprint, &settings),
            fingerprint,
            settings,
            plan.phase_summaries(),
        )
        .map_err(to_python_error)?;
        Ok(Self::new(plan, state, &inputs))
    }

    /// Open an existing state after validating its owner-only secret files
    #[staticmethod]
    fn open(mnemonic_file: &str, words_file: &str, state_dir: &str) -> PyResult<Self> {
        let existing =
            RecoveryState::open_existing(Path::new(state_dir)).map_err(to_python_error)?;
        let manifest = existing.manifest().clone();
        let inputs = load_inputs(Path::new(mnemonic_file), Path::new(words_file))
            .map_err(to_python_error)?;
        let plan = RecoveryPlan::compile(&inputs.written_words, manifest.settings.clone())
            .map_err(to_python_error)?;
        let state = RecoveryState::open_or_create(
            Path::new(state_dir),
            recovery_spec_hash(&inputs, manifest.target_fingerprint, &manifest.settings),
            manifest.target_fingerprint,
            manifest.settings,
            plan.phase_summaries(),
        )
        .map_err(to_python_error)?;
        Ok(Self::new(plan, state, &inputs))
    }

    /// Transfer the normalized mnemonic to a mutable host array exactly once
    fn take_mnemonic_bytes(&mut self, py: Python<'_>) -> PyResult<Py<PyArray1<u8>>> {
        if self.mnemonic_taken {
            return Err(RecoveryError::new_err(
                "mnemonic material was already transferred",
            ));
        }
        self.mnemonic_taken = true;
        let bytes = self.mnemonic.as_slice().to_vec();
        self.mnemonic.zeroize();
        Ok(Array1::from_vec(bytes).into_pyarray(py).unbind())
    }

    /// Exact phase counts for the shared v2 plan
    fn phase_summaries(&self) -> Vec<(String, u128)> {
        self.plan
            .phase_summaries()
            .into_iter()
            .map(|summary| (summary.phase.to_string(), summary.count))
            .collect()
    }

    /// Ranked nearest-word suggestions and edit distances
    fn neighbor_suggestions(&self) -> Vec<(String, Vec<(String, usize)>)> {
        self.plan
            .neighbor_suggestions()
            .iter()
            .map(|suggestion| {
                (
                    suggestion.written.clone(),
                    suggestion
                        .neighbors
                        .iter()
                        .map(|neighbor| (neighbor.word.clone(), neighbor.distance))
                        .collect(),
                )
            })
            .collect()
    }

    /// Immutable settings defining the candidate space
    fn settings(&self) -> (usize, usize, usize, usize, bool) {
        let settings = self.plan.settings();
        (
            settings.neighbors_per_word,
            settings.max_replacements,
            settings.local_swap_radius,
            settings.max_passphrase_bytes,
            settings.lowercase_already_tried,
        )
    }

    /// Target Coldcard fingerprint in display order
    fn target_fingerprint(&self) -> String {
        self.state.manifest().target_fingerprint.to_string()
    }

    /// Number of unique candidates durably verified
    fn completed(&self) -> u128 {
        self.state.runtime().cursor.completed
    }

    /// Whether manual verification blocks further candidate generation
    fn has_pending_matches(&self) -> bool {
        self.state.has_pending_matches()
    }

    /// Pending matches with intentional secret output for the CLI
    fn pending_matches(&self) -> Vec<(String, String, String, Vec<String>)> {
        self.state
            .runtime()
            .matches
            .iter()
            .filter(|record| record.status == MatchStatus::Pending)
            .map(|record| {
                (
                    record.id.0.clone(),
                    record.phase.to_string(),
                    record.passphrase.clone(),
                    record.words.clone(),
                )
            })
            .collect()
    }

    /// Prepare one fixed-width batch and retain its candidate/cursor association
    fn prepare_batch(
        &mut self,
        py: Python<'_>,
        through: &str,
        batch_size: usize,
    ) -> PyResult<Option<PreparedBatch>> {
        if batch_size == 0 {
            return Err(RecoveryError::new_err("batch size must be positive"));
        }
        if self.pending.is_some() {
            return Err(RecoveryError::new_err(
                "the previous recovery batch is still pending",
            ));
        }
        if self.state.has_pending_matches() {
            return Err(to_python_error(RecoverError::PendingMatches));
        }
        let through = parse_phase(through)?;

        loop {
            let mut cursor = self.state.cursor();
            let candidates = self
                .plan
                .next_batch(&mut cursor, through, batch_size)
                .map_err(to_python_error)?;
            if candidates.is_empty() {
                self.state
                    .complete_chunk(cursor, Vec::new())
                    .map_err(to_python_error)?;
                return Ok(None);
            }
            let active = candidates
                .into_iter()
                .filter(|candidate| !self.state.is_rejected(candidate.id()))
                .collect::<Vec<_>>();
            if active.is_empty() {
                self.state
                    .complete_chunk(cursor, Vec::new())
                    .map_err(to_python_error)?;
                continue;
            }

            self.next_token = self.next_token.wrapping_add(1).max(1);
            let token = self.next_token;
            let prepared = prepared_batch(py, token, batch_size, &active)?;
            self.pending = Some(PendingBatch {
                token,
                cursor,
                candidates: active,
            });
            return Ok(Some(prepared));
        }
    }

    /// Verify all real rows and atomically commit the associated cursor and matches
    fn complete_batch(
        &mut self,
        token: u64,
        seeds: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<BatchCompletion> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| RecoveryError::new_err("no recovery batch is pending"))?;
        if pending.token != token {
            return Err(RecoveryError::new_err(
                "seed batch token does not match the pending recovery batch",
            ));
        }
        let seed_batch = seed_batch(&seeds, pending.candidates.len())?;
        let indices =
            matching_candidate_indices(&seed_batch, self.state.manifest().target_fingerprint)
                .map_err(to_python_error)?;
        let matches = indices
            .into_iter()
            .map(|index| MatchRecord::pending(&pending.candidates[index]))
            .collect::<Vec<_>>();
        let match_count = matches.len();
        self.state
            .complete_chunk(pending.cursor.clone(), matches)
            .map_err(to_python_error)?;
        let checked = pending.candidates.len();
        let completed = pending.cursor.completed;
        self.pending = None;
        Ok(BatchCompletion {
            checked,
            matches: match_count,
            completed,
        })
    }

    /// Verify a seed sample without changing durable recovery progress
    fn fingerprint_batch(&self, seeds: PyReadonlyArray2<'_, u8>) -> PyResult<Vec<usize>> {
        let batch = seed_batch(&seeds, seeds.shape()[0])?;
        matching_candidate_indices(&batch, self.state.manifest().target_fingerprint)
            .map_err(to_python_error)
    }

    /// Generate a deterministic benchmark sample without changing state
    fn sample_batch(&self, py: Python<'_>, batch_size: usize) -> PyResult<PreparedBatch> {
        if batch_size == 0 {
            return Err(RecoveryError::new_err("batch size must be positive"));
        }
        let mut cursor = CandidateCursor::default();
        let candidates = self
            .plan
            .next_batch(&mut cursor, SearchPhase::WrittenCase, batch_size)
            .map_err(to_python_error)?;
        if candidates.is_empty() {
            return Err(RecoveryError::new_err(
                "recovery plan produced no benchmark candidates",
            ));
        }
        prepared_batch(py, 0, batch_size, &candidates)
    }

    /// Enumerate metadata from an optional serialized cursor for differential tests
    #[pyo3(signature = (through, limit, cursor_json=None))]
    fn enumerate_candidates(
        &self,
        through: &str,
        limit: usize,
        cursor_json: Option<&str>,
    ) -> PyResult<(Vec<CandidateRecord>, String)> {
        let through = parse_phase(through)?;
        let mut cursor = cursor_json.map_or_else(
            || Ok(CandidateCursor::default()),
            |json| {
                serde_json::from_str(json)
                    .map_err(|_| RecoveryError::new_err("invalid candidate cursor JSON"))
            },
        )?;
        let candidates = self
            .plan
            .next_batch(&mut cursor, through, limit)
            .map_err(to_python_error)?;
        let records = candidates
            .into_iter()
            .map(|candidate| {
                (
                    candidate.id().0.clone(),
                    candidate.phase().to_string(),
                    candidate.passphrase().to_owned(),
                    candidate.words().to_vec(),
                )
            })
            .collect();
        let cursor = serde_json::to_string(&cursor)
            .map_err(|_| RecoveryError::new_err("failed to serialize candidate cursor"))?;
        Ok((records, cursor))
    }
}

impl RecoverySession {
    fn new(plan: RecoveryPlan, state: RecoveryState, inputs: &RecoveryInputs) -> Self {
        Self {
            plan,
            state,
            mnemonic: Zeroizing::new(inputs.mnemonic.expose().as_bytes().to_vec()),
            mnemonic_taken: false,
            pending: None,
            next_token: 0,
        }
    }
}

fn prepared_batch(
    py: Python<'_>,
    token: u64,
    batch_size: usize,
    candidates: &[Candidate],
) -> PyResult<PreparedBatch> {
    let mut bytes = vec![0_u8; batch_size * DEFAULT_MAX_PASSPHRASE_BYTES];
    let mut lengths = vec![0_u16; batch_size];
    for (index, candidate) in candidates.iter().enumerate() {
        let passphrase = candidate.passphrase().as_bytes();
        if passphrase.len() > DEFAULT_MAX_PASSPHRASE_BYTES {
            return Err(RecoveryError::new_err(
                "candidate exceeds the supported passphrase length",
            ));
        }
        let start = index * DEFAULT_MAX_PASSPHRASE_BYTES;
        bytes[start..start + passphrase.len()].copy_from_slice(passphrase);
        lengths[index] = passphrase
            .len()
            .try_into()
            .map_err(|_| RecoveryError::new_err("candidate length is out of range"))?;
    }
    let candidate_bytes = Array2::from_shape_vec((batch_size, DEFAULT_MAX_PASSPHRASE_BYTES), bytes)
        .map_err(|_| RecoveryError::new_err("failed to shape candidate batch"))?
        .into_pyarray(py)
        .unbind();
    let lengths = Array1::from_vec(lengths).into_pyarray(py).unbind();
    Ok(PreparedBatch {
        token,
        count: candidates.len(),
        candidate_bytes,
        lengths,
    })
}

fn seed_batch(seeds: &PyReadonlyArray2<'_, u8>, count: usize) -> PyResult<SeedBatch> {
    let shape = seeds.shape();
    if shape.len() != 2 || shape[0] < count || shape[1] != 64 {
        return Err(RecoveryError::new_err(
            "seed matrix must have shape (batch, 64)",
        ));
    }
    let array = seeds.as_array();
    let mut output = Vec::with_capacity(count);
    for row in array.rows().into_iter().take(count) {
        let mut seed = [0_u8; 64];
        seed.copy_from_slice(
            row.as_slice()
                .ok_or_else(|| RecoveryError::new_err("seed matrix must be contiguous"))?,
        );
        output.push(seed);
    }
    Ok(SeedBatch::new(output))
}

fn parse_phase(value: &str) -> PyResult<SearchPhase> {
    match value {
        "written-lower" => Ok(SearchPhase::WrittenLower),
        "written-case" => Ok(SearchPhase::WrittenCase),
        "neighbor-1-lower" => Ok(SearchPhase::Neighbor1Lower),
        "neighbor-2-lower" => Ok(SearchPhase::Neighbor2Lower),
        "neighbor-1-case" => Ok(SearchPhase::Neighbor1Case),
        "neighbor-2-case" => Ok(SearchPhase::Neighbor2Case),
        _ => Err(RecoveryError::new_err("unknown recovery phase")),
    }
}

fn to_python_error(error: RecoverError) -> PyErr {
    let message = match error {
        RecoverError::InvalidMnemonic(_) => "invalid English BIP39 mnemonic".to_owned(),
        RecoverError::InvalidWrittenWord { line, .. } => {
            format!("written word on line {line} must contain ASCII letters only")
        }
        other => other.to_string(),
    };
    RecoveryError::new_err(message)
}

/// Mark a pending shared-state match as a rejected four-byte collision
#[pyfunction]
fn reject_match(state_dir: &str, match_id: &str) -> PyResult<()> {
    let mut state = RecoveryState::open_existing(Path::new(state_dir)).map_err(to_python_error)?;
    state.reject_match(match_id).map_err(to_python_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RecoveryError", module.py().get_type::<RecoveryError>())?;
    module.add_class::<PreparedBatch>()?;
    module.add_class::<BatchCompletion>()?;
    module.add_class::<RecoverySession>()?;
    module.add_function(wrap_pyfunction!(reject_match, module)?)?;
    Ok(())
}
