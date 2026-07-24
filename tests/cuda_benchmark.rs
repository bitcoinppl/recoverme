use std::num::NonZeroUsize;

use bip32::{Prefix, XPrv};
use bip39::{Language, Mnemonic};
use recoverme::{
    benchmark::{benchmark_sustained_pipeline, SustainedBenchmarkConfig, SustainedBenchmarkResult},
    domain::{BackendConfiguration, BackendKind, OrderMode, SpacingMode},
    MasterXpubTarget, RecoverError, RecoverySettings, SearchPhase, SecretMnemonic,
    TargetFingerprint, VerificationTarget, WrittenWords,
};

const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
const PUBLIC_TEST_PASSPHRASE: &str = "alphabriskcactusdaringeagerfabricgadget";
const PUBLIC_TEST_WORDS: [&str; 7] = [
    "alpha", "brisk", "cactus", "daring", "eager", "fabric", "gadget",
];
const TUNING_BATCH_SIZES: [usize; 3] = [262_144, 524_288, 1_048_576];
const TUNING_WORKGROUP_SIZES: [u32; 3] = [64, 128, 256];
const TUNING_CPU_SHARES: [u8; 4] = [2, 5, 10, 20];
const TUNING_REPETITIONS: usize = 2;
const SUSTAINED_REPETITIONS: usize = 5;

#[test]
#[ignore = "requires an NVIDIA RTX 4090 CUDA device"]
fn cuda_sustained_xfp_and_xpub_throughput() {
    let secret = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
    let target = public_master_xpub_target();
    let fingerprint_target = VerificationTarget::Fingerprint(target.fingerprint());
    let words = WrittenWords::new(PUBLIC_TEST_WORDS.map(str::to_owned).to_vec()).unwrap();
    let settings = RecoverySettings {
        max_replacements: 0,
        order: OrderMode::Permuted,
        spacing: SpacingMode::Concatenated,
        ..RecoverySettings::default()
    };
    let plan = recoverme::search::RecoveryPlan::compile(&words, settings).unwrap();

    let xfp = autotune_and_benchmark(&plan, &secret, &fingerprint_target).unwrap();
    let xpub = autotune_and_benchmark(&plan, &secret, &target).unwrap();

    for result in [xfp.cuda, xfp.hybrid, xpub.cuda, xpub.hybrid] {
        assert_eq!(result.matches, 1);
        assert!(result.sustained_checks_per_second.is_finite());
        assert!(result.sustained_checks_per_second > 0.0);
    }
}

struct ModeResults {
    cuda: SustainedBenchmarkResult,
    hybrid: SustainedBenchmarkResult,
}

fn autotune_and_benchmark(
    plan: &recoverme::search::RecoveryPlan,
    secret: &SecretMnemonic,
    target: &VerificationTarget,
) -> Result<ModeResults, RecoverError> {
    let mut cuda_tuning = Vec::new();
    for batch_size in TUNING_BATCH_SIZES {
        for workgroup_size in TUNING_WORKGROUP_SIZES {
            let configuration = BackendConfiguration::cube(batch_size, workgroup_size)?;
            let result = run_benchmark(
                plan,
                secret,
                target,
                BackendKind::Cuda,
                configuration,
                TUNING_REPETITIONS,
            )?;
            print_result("tuning", &result);
            cuda_tuning.push(result);
        }
    }
    let selected_cuda = select_fastest(cuda_tuning);
    let cuda_configuration = BackendConfiguration::cube(
        selected_cuda.batch_size,
        selected_cuda
            .workgroup_size
            .expect("CUDA tuning records a workgroup size"),
    )?;

    let mut hybrid_tuning = Vec::new();
    for cpu_share in TUNING_CPU_SHARES {
        let configuration = BackendConfiguration::hybrid(
            selected_cuda.batch_size,
            selected_cuda
                .workgroup_size
                .expect("CUDA tuning records a workgroup size"),
            cpu_share,
        )?;
        let result = run_benchmark(
            plan,
            secret,
            target,
            BackendKind::CudaHybrid,
            configuration,
            TUNING_REPETITIONS,
        )?;
        print_result("tuning", &result);
        hybrid_tuning.push(result);
    }
    let selected_hybrid = select_fastest(hybrid_tuning);
    let hybrid_configuration = BackendConfiguration::hybrid(
        selected_hybrid.batch_size,
        selected_hybrid
            .workgroup_size
            .expect("CUDA hybrid tuning records a workgroup size"),
        selected_hybrid
            .cpu_share_percent
            .expect("CUDA hybrid tuning records a CPU share"),
    )?;

    let cuda = run_benchmark(
        plan,
        secret,
        target,
        BackendKind::Cuda,
        cuda_configuration,
        SUSTAINED_REPETITIONS,
    )?;
    print_result("sustained", &cuda);
    let hybrid = run_benchmark(
        plan,
        secret,
        target,
        BackendKind::CudaHybrid,
        hybrid_configuration,
        SUSTAINED_REPETITIONS,
    )?;
    print_result("sustained", &hybrid);

    Ok(ModeResults { cuda, hybrid })
}

fn run_benchmark(
    plan: &recoverme::search::RecoveryPlan,
    secret: &SecretMnemonic,
    target: &VerificationTarget,
    backend: BackendKind,
    configuration: BackendConfiguration,
    repetitions: usize,
) -> Result<SustainedBenchmarkResult, RecoverError> {
    benchmark_sustained_pipeline(
        plan,
        secret,
        target,
        backend,
        SustainedBenchmarkConfig {
            sample_size: NonZeroUsize::new(configuration.batch_size().get())
                .expect("backend configurations use nonzero batch sizes"),
            repetitions: NonZeroUsize::new(repetitions).expect("benchmark repetitions are nonzero"),
            configuration,
            through: SearchPhase::WrittenCase,
        },
    )
}

fn select_fastest(results: Vec<SustainedBenchmarkResult>) -> SustainedBenchmarkResult {
    results
        .into_iter()
        .max_by(|left, right| {
            left.sustained_checks_per_second
                .total_cmp(&right.sustained_checks_per_second)
        })
        .expect("autotuning measures at least one configuration")
}

fn public_master_xpub_target() -> VerificationTarget {
    let mnemonic = Mnemonic::parse_in(Language::English, PUBLIC_TEST_MNEMONIC).unwrap();
    let master = XPrv::new(mnemonic.to_seed(PUBLIC_TEST_PASSPHRASE)).unwrap();
    let master_xpub =
        MasterXpubTarget::parse(&master.public_key().to_string(Prefix::XPUB)).unwrap();
    let fingerprint = hex::encode(master_xpub.fingerprint())
        .parse::<TargetFingerprint>()
        .unwrap();
    VerificationTarget::new(fingerprint, Some(master_xpub)).unwrap()
}

fn print_result(stage: &str, result: &SustainedBenchmarkResult) {
    println!(
        "GPUQ_BENCHMARK stage={stage} mode={} backend={} device={} sample={} repetitions={} batch={} workgroup={} cpu_share={} sustained_checks_per_second={:.1} median_checks_per_second={:.1} minimum_checks_per_second={:.1} maximum_checks_per_second={:.1} matches={} iteration_milliseconds={:?} warmup_excluded=true",
        result.verification_mode,
        result.backend,
        result.device,
        result.sample_size,
        result.repetitions,
        result.batch_size,
        result
            .workgroup_size
            .map_or_else(|| "n/a".into(), |size| size.to_string()),
        result
            .cpu_share_percent
            .map_or_else(|| "n/a".into(), |share| format!("{share}%")),
        result.sustained_checks_per_second,
        result.median_checks_per_second,
        result.minimum_checks_per_second,
        result.maximum_checks_per_second,
        result.matches,
        result.iteration_milliseconds
    );
}
