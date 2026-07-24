use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

use bip39::Language;
use clap::{Args, Parser, Subcommand};
use color_eyre::eyre::{Result, WrapErr};
use recoverme::{
    backend::{available_backends, resolve_backend},
    domain::{
        BackendConfiguration, BackendKind, OrderMode, RecoverySettings, SearchPhase, SpacingMode,
        TargetFingerprint, VerificationTarget,
    },
    engine::{benchmark_backend, measure_backend_with_config, run_recovery, RunOutcome},
    input::{
        load_inputs, load_inputs_from_env, load_inputs_with_recipe, load_master_xpub,
        load_target_fingerprint_from_env, recovery_spec_hash_for_target, RecoveryInputs,
    },
    search::{expected_xfp_collisions, xfp_collision_probability, RecoveryPlan},
    state::{MatchStatus, RecoveryState},
};

#[derive(Debug, Parser)]
#[command(author, version, about, arg_required_else_help = true)]
struct Cli {
    /// Owner-only TOML file providing command defaults
    #[arg(long, global = true, env = "RECOVERME_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate inputs, compile the candidate space, and create durable state
    Plan(PlanArgs),
    /// Measure one or every compiled seed-derivation backend
    Benchmark(SessionArgs),
    /// Execute or resume the ranked search through an explicit phase
    Run(RunArgs),
    /// Reject a pending match after manual Coldcard verification
    RejectMatch {
        /// Durable recovery state directory
        #[arg(long, env = "RECOVERME_STATE_DIR")]
        state_dir: PathBuf,
        /// Candidate identifier printed when the match was found
        #[arg(long)]
        match_id: String,
    },
}

#[derive(Debug, Args)]
struct SecretInputs {
    /// Owner-only mnemonic file; omit file flags to use scoped environment inputs
    #[arg(long, env = "RECOVERME_MNEMONIC_FILE")]
    mnemonic_file: Option<PathBuf>,
    /// Owner-only written-words file; omit both file flags to use the environment
    #[arg(long, env = "RECOVERME_WORDS_FILE", conflicts_with = "recipe_file")]
    words_file: Option<PathBuf>,
    /// Owner-only advanced recovery recipe file
    #[arg(long, env = "RECOVERME_RECIPE_FILE", conflicts_with = "words_file")]
    recipe_file: Option<PathBuf>,
}

impl SecretInputs {
    fn load(&self) -> Result<RecoveryInputs> {
        match (&self.mnemonic_file, &self.words_file, &self.recipe_file) {
            (Some(mnemonic), Some(words), None) => Ok(load_inputs(mnemonic, words)?),
            (Some(mnemonic), None, Some(recipe)) => Ok(load_inputs_with_recipe(mnemonic, recipe)?),
            (None, None, None) => Ok(load_inputs_from_env()?),
            _ => Err(recoverme::error::RecoverError::InvalidSetting(
                "mnemonic-file must be paired with words-file or recipe-file".into(),
            )
            .into()),
        }
    }
}

#[derive(Debug, Args)]
struct PlanArgs {
    #[command(flatten)]
    secrets: SecretInputs,
    /// Target XFP; omit to read `XFP` from the environment
    #[arg(long, env = "RECOVERME_FINGERPRINT")]
    fingerprint: Option<TargetFingerprint>,
    /// Owner-only file containing the depth-zero master extended public key
    #[arg(long, env = "RECOVERME_MASTER_XPUB_FILE")]
    master_xpub_file: Option<PathBuf>,
    /// Durable recovery state directory
    #[arg(long, env = "RECOVERME_STATE_DIR")]
    state_dir: PathBuf,
    /// BIP39 neighbors retained for each written word
    #[arg(long, env = "RECOVERME_NEIGHBORS", default_value_t = 3)]
    neighbors: usize,
    /// Maximum number of nearest-word substitutions
    #[arg(long, env = "RECOVERME_MAX_REPLACEMENTS", default_value_t = 2)]
    max_replacements: usize,
    /// Exclude lowercase-only candidates completed by the earlier CPU run
    #[arg(long, env = "RECOVERME_LOWERCASE_ALREADY_TRIED")]
    lowercase_already_tried: bool,
    /// Candidate token order
    #[arg(long, env = "RECOVERME_ORDER", value_enum, default_value_t = OrderMode::Permuted)]
    order: OrderMode,
    /// Candidate spacing strategy
    #[arg(long, env = "RECOVERME_SPACING", value_enum, default_value_t = SpacingMode::Concatenated)]
    spacing: SpacingMode,
    /// Exclude the all-concatenated pattern as previously exhausted
    #[arg(long, env = "RECOVERME_CONCATENATED_ALREADY_TRIED")]
    concatenated_already_tried: bool,
}

#[derive(Debug, Args)]
struct SessionArgs {
    #[command(flatten)]
    secrets: SecretInputs,
    /// Durable recovery state directory
    #[arg(long, env = "RECOVERME_STATE_DIR")]
    state_dir: PathBuf,
    /// Backend to benchmark, or auto for every compiled backend
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,
    /// Number of candidates in the benchmark sample
    #[arg(long, default_value_t = 65_536)]
    sample_size: usize,
    /// Sweep production batch and workgroup sizes and retain the fastest result
    #[arg(long, env = "RECOVERME_AUTOTUNE")]
    autotune: bool,
}

#[derive(Debug, Args)]
struct RunArgs {
    #[command(flatten)]
    secrets: SecretInputs,
    /// Durable recovery state directory
    #[arg(long, env = "RECOVERME_STATE_DIR")]
    state_dir: PathBuf,
    /// Last phase authorized for this run
    #[arg(long)]
    through: SearchPhase,
    /// Seed-derivation backend
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,
    /// Skip the final count and ETA confirmation
    #[arg(long)]
    yes: bool,
}

fn main() -> Result<()> {
    recoverme::config::apply_config_defaults()?;
    pretty_env_logger::init();
    color_eyre::install()?;

    let cli = Cli::parse();
    let _config = cli.config;
    match cli.command {
        Command::Plan(args) => plan(args),
        Command::Benchmark(args) => benchmark(args),
        Command::Run(args) => run(args),
        Command::RejectMatch {
            state_dir,
            match_id,
        } => reject_match(&state_dir, &match_id),
    }
}

fn plan(args: PlanArgs) -> Result<()> {
    let inputs = args.secrets.load()?;
    let settings = RecoverySettings {
        neighbors_per_word: args.neighbors,
        max_replacements: args.max_replacements,
        lowercase_already_tried: args.lowercase_already_tried,
        order: args.order,
        spacing: args.spacing,
        concatenated_already_tried: args.concatenated_already_tried,
        ..RecoverySettings::default()
    };
    let recovery_plan = RecoveryPlan::compile_recipe(&inputs.recipe, settings.clone())?;
    let fingerprint = args
        .fingerprint
        .map_or_else(load_target_fingerprint_from_env, Ok)?;
    let target = VerificationTarget::new(
        fingerprint,
        args.master_xpub_file
            .as_deref()
            .map(load_master_xpub)
            .transpose()?,
    )?;
    let spec_hash = recovery_spec_hash_for_target(&inputs, &target, &settings);
    let mut state = RecoveryState::open_or_create_for_target(
        &args.state_dir,
        spec_hash,
        target,
        settings,
        recovery_plan.phase_summaries(),
    )?;

    if state.latest_benchmark(BackendKind::Cpu).is_none() {
        let benchmark = benchmark_backend(
            &recovery_plan,
            &mut state,
            &inputs.mnemonic,
            BackendKind::Cpu,
            4_096,
        )?;
        println!(
            "CPU baseline: {:.1} complete checks/s ({})",
            benchmark.checks_per_second, benchmark.device
        );
    }
    print_plan(&recovery_plan, &state);
    println!("State: {}", args.state_dir.display());
    Ok(())
}

fn benchmark(args: SessionArgs) -> Result<()> {
    let (inputs, plan, mut state) = load_session(&args.state_dir, &args.secrets)?;
    let backends = if args.backend == BackendKind::Auto {
        available_backends()
    } else {
        vec![args.backend]
    };

    for backend in backends {
        if args.autotune {
            autotune_backend(
                &plan,
                &mut state,
                &inputs.mnemonic,
                backend,
                args.sample_size,
            )?;
        } else {
            let record = benchmark_backend(
                &plan,
                &mut state,
                &inputs.mnemonic,
                backend,
                args.sample_size,
            )?;
            print_benchmark(&record);
        }
    }
    Ok(())
}

fn autotune_backend(
    plan: &RecoveryPlan,
    state: &mut RecoveryState,
    mnemonic: &recoverme::domain::SecretMnemonic,
    backend: BackendKind,
    sample_size: usize,
) -> Result<()> {
    const REPETITIONS: usize = 3;

    let mut accelerator_batch_sizes = vec![
        sample_size.saturating_div(4).max(1),
        sample_size.max(1),
        sample_size.saturating_mul(2).max(1),
    ];
    accelerator_batch_sizes.sort_unstable();
    accelerator_batch_sizes.dedup();
    let configurations = match backend {
        BackendKind::Cpu => {
            let base = rayon::current_num_threads().max(1) * 128;
            [base, base * 4, base * 16]
                .into_iter()
                .map(BackendConfiguration::cpu)
                .collect::<Result<Vec<_>, _>>()?
        }
        BackendKind::CubeCpu | BackendKind::Metal | BackendKind::Cuda => accelerator_batch_sizes
            .iter()
            .copied()
            .flat_map(|batch_size| {
                [32, 64, 128].into_iter().map(move |workgroup_size| {
                    BackendConfiguration::cube(batch_size, workgroup_size)
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        BackendKind::Hybrid => accelerator_batch_sizes
            .iter()
            .copied()
            .flat_map(|batch_size| {
                [32, 64, 128].into_iter().map(move |workgroup_size| {
                    BackendConfiguration::hybrid(batch_size, workgroup_size, 35)
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        unavailable => {
            return Err(
                recoverme::error::RecoverError::BackendUnavailable(unavailable.to_string()).into(),
            )
        }
    };
    let mut best = None;
    for configuration in configurations {
        let mut records = Vec::with_capacity(REPETITIONS);
        for _ in 0..REPETITIONS {
            let record = measure_backend_with_config(
                plan,
                state,
                mnemonic,
                backend,
                configuration.batch_size().get(),
                configuration,
            )?;
            print_benchmark(&record);
            records.push(record);
        }
        let record = median_benchmark(records);
        println!(
            "Median {backend}: batch={}, workgroup={}, {:.1} sustained checks/s",
            record.batch_size,
            record
                .workgroup_size
                .map_or_else(|| "n/a".into(), |size| size.to_string()),
            record.checks_per_second
        );
        if best
            .as_ref()
            .is_none_or(|best: &recoverme::state::BenchmarkRecord| {
                record.checks_per_second > best.checks_per_second
            })
        {
            best = Some(record);
        }
    }
    let best = best.expect("every backend has benchmark configurations");
    state.record_benchmark(best.clone())?;
    println!(
        "Selected {backend}: batch={}, workgroup={}, {:.1} complete checks/s",
        best.batch_size,
        best.workgroup_size
            .map_or_else(|| "n/a".into(), |size| size.to_string()),
        best.checks_per_second
    );
    Ok(())
}

fn median_benchmark(
    mut records: Vec<recoverme::state::BenchmarkRecord>,
) -> recoverme::state::BenchmarkRecord {
    records.sort_by(|left, right| left.checks_per_second.total_cmp(&right.checks_per_second));
    records.swap_remove(records.len() / 2)
}

fn print_benchmark(record: &recoverme::state::BenchmarkRecord) {
    println!(
        "{}: {:.1} seeds/s, {:.1} verifications/s, {:.1} sustained checks/s, {:.1} candidates/s, sample={}, batch={}, workgroup={}, cpu-share={}, device={}",
        record.backend,
        record.seeds_per_second,
        record.verification_per_second,
        record.checks_per_second,
        record.generation_per_second,
        record.candidates,
        record.batch_size,
        record
            .workgroup_size
            .map_or_else(|| "n/a".into(), |size| size.to_string()),
        record
            .cpu_share_percent
            .map_or_else(|| "n/a".into(), |share| format!("{share}%")),
        record.device
    );
}

fn quick_sample_size(backend: BackendKind) -> usize {
    match backend {
        BackendKind::Cpu | BackendKind::CubeCpu => 4_096,
        BackendKind::Metal | BackendKind::Hybrid | BackendKind::Cuda => 16_384,
        BackendKind::Auto => 4_096,
    }
}

fn run(args: RunArgs) -> Result<()> {
    let (inputs, plan, mut state) = load_session(&args.state_dir, &args.secrets)?;
    if args.backend == BackendKind::Auto {
        for backend in available_backends() {
            if state.latest_benchmark(backend).is_some() {
                continue;
            }
            let record = benchmark_backend(
                &plan,
                &mut state,
                &inputs.mnemonic,
                backend,
                quick_sample_size(backend),
            )?;
            println!(
                "Measured {backend}: {:.1} complete checks/s",
                record.checks_per_second
            );
        }
    }
    let backend = resolve_backend(args.backend, &state);
    if state.latest_benchmark(backend).is_none() {
        let record = benchmark_backend(
            &plan,
            &mut state,
            &inputs.mnemonic,
            backend,
            quick_sample_size(backend),
        )?;
        println!(
            "Measured {backend}: {:.1} complete checks/s",
            record.checks_per_second
        );
    }

    let total = plan.count_through(args.through)?;
    let completed = state.runtime().cursor.completed.min(total);
    let remaining = total - completed;
    let rate = state
        .latest_benchmark(backend)
        .map(|benchmark| benchmark.checks_per_second)
        .unwrap_or(0.0);
    println!("Backend: {backend}");
    println!("Authorized through: {}", args.through);
    println!("Remaining candidates: {}", format_count(remaining));
    if rate > 0.0 {
        println!(
            "Estimated remaining time: {}",
            format_duration(remaining as f64 / rate)
        );
    }
    println!(
        "Expected random four-byte XFP hits in authorized space: {:.4}",
        expected_xfp_collisions(total)
    );
    let target = state.verification_target()?;
    if target.master_xpub().is_some() {
        println!("Master XPUB filtering is enabled; full public-key confirmation is still followed by manual Coldcard verification");
    }

    if !args.yes && !confirm("Start or resume this search? [y/N] ")? {
        println!("Cancelled");
        return Ok(());
    }

    let stop = Arc::new(AtomicBool::new(false));
    let handler_stop = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        handler_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    })
    .wrap_err("failed to install Ctrl-C handler")?;

    match run_recovery(
        &plan,
        &mut state,
        &inputs.mnemonic,
        &target,
        backend,
        args.through,
        stop,
    )? {
        RunOutcome::Exhausted => {
            println!("Authorized phases exhausted without a pending XFP match")
        }
        RunOutcome::Interrupted => println!("Stopped cleanly after the last completed chunk"),
        RunOutcome::MatchesFound(count) => {
            println!("Found {count} wallet candidate(s); verify each manually on the Coldcard");
            for record in state
                .runtime()
                .matches
                .iter()
                .filter(|record| record.status == MatchStatus::Pending)
            {
                println!("Match ID: {}", record.id);
                println!("Exact passphrase: {}", record.passphrase);
                println!("Readable words: {}", record.words.join(" "));
            }
        }
    }
    Ok(())
}

fn reject_match(state_dir: &Path, match_id: &str) -> Result<()> {
    let mut state = RecoveryState::open_existing(state_dir)?;
    state.reject_match(match_id)?;
    println!("Rejected match {match_id}; the next run will continue from its checkpoint");
    Ok(())
}

fn load_session(
    state_dir: &Path,
    secrets: &SecretInputs,
) -> Result<(RecoveryInputs, RecoveryPlan, RecoveryState)> {
    let existing = RecoveryState::open_existing(state_dir)?;
    let manifest = existing.manifest().clone();
    let target = existing.verification_target()?;
    let inputs = secrets.load()?;
    let plan = RecoveryPlan::compile_recipe(&inputs.recipe, manifest.settings.clone())?;
    let spec_hash = recovery_spec_hash_for_target(&inputs, &target, &manifest.settings);
    let state = RecoveryState::open_or_create_for_target(
        state_dir,
        spec_hash,
        target,
        manifest.settings,
        plan.phase_summaries(),
    )?;
    Ok((inputs, plan, state))
}

fn print_plan(plan: &RecoveryPlan, state: &RecoveryState) {
    let bip39_words = Language::English.word_list();
    println!("Nearest BIP39 words:");
    for suggestion in plan.neighbor_suggestions() {
        let warning = if bip39_words.contains(&suggestion.written.as_str()) {
            ""
        } else {
            " (not itself a BIP39 word)"
        };
        let neighbors = suggestion
            .neighbors
            .iter()
            .map(|neighbor| format!("{} [d={}]", neighbor.word, neighbor.distance))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {}{}: {}", suggestion.written, warning, neighbors);
    }

    let rate = state
        .latest_benchmark(BackendKind::Cpu)
        .map(|benchmark| benchmark.checks_per_second)
        .unwrap_or(0.0);
    if plan.settings().lowercase_already_tried {
        println!("Lowercase-only phases: excluded as previously completed");
    }
    println!("Search phases:");
    for summary in plan.phase_summaries() {
        let probability = xfp_collision_probability(summary.count) * 100.0;
        if rate > 0.0 {
            println!(
                "  {:<20} {:>20}  ETA {:>12}  XFP collision {:>8.4}%",
                summary.phase,
                format_count(summary.count),
                format_duration(summary.count as f64 / rate),
                probability
            );
        } else {
            println!(
                "  {:<20} {:>20}  XFP collision {:>8.4}%",
                summary.phase,
                format_count(summary.count),
                probability
            );
        }
    }
}

fn confirm(prompt: &str) -> Result<bool> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn format_count(value: u128) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "over numeric range".into();
    }
    if seconds < 60.0 {
        return format!("{seconds:.1}s");
    }
    if seconds < 3_600.0 {
        return format!("{:.1}m", seconds / 60.0);
    }
    if seconds < 86_400.0 {
        return format!("{:.1}h", seconds / 3_600.0);
    }
    if seconds < 31_557_600.0 {
        return format!("{:.1}d", seconds / 86_400.0);
    }
    format!("{:.2}y", seconds / 31_557_600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_accepts_environment_backed_inputs() {
        let cli = Cli::try_parse_from(["recoverme", "plan", "--state-dir", "state"]);

        assert!(cli.is_ok());
    }

    #[test]
    fn secret_file_arguments_are_validated_together() {
        let cli = Cli::try_parse_from([
            "recoverme",
            "plan",
            "--state-dir",
            "state",
            "--mnemonic-file",
            "mnemonic.txt",
        ]);

        let cli = cli.unwrap();
        let Command::Plan(args) = cli.command else {
            panic!("expected plan command");
        };
        assert!(args.secrets.load().is_err());
    }
}
