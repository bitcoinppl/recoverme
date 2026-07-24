use std::num::{NonZeroU32, NonZeroUsize};

use bip32::{Prefix, XPrv};
use bip39::{Language, Mnemonic};
use recoverme::{
    benchmark::{benchmark_sustained_pipeline, SustainedBenchmarkConfig, SustainedBenchmarkResult},
    domain::{OrderMode, SpacingMode},
    MasterXpubTarget, RecoverySettings, SearchPhase, SecretMnemonic, TargetFingerprint,
    VerificationTarget, WrittenWords,
};

const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
const PUBLIC_TEST_PASSPHRASE: &str = "alphabriskcactusdaringeagerfabricgadget";
const PUBLIC_TEST_WORDS: [&str; 7] = [
    "alpha", "brisk", "cactus", "daring", "eager", "fabric", "gadget",
];
const TUNING_BATCH_SIZES: [usize; 4] = [65_536, 131_072, 262_144, 524_288];
const TUNING_REPETITIONS: usize = 3;
const SUSTAINED_REPETITIONS: usize = 5;
const WORKGROUP_SIZE: u32 = 128;

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

    let xfp = autotune_and_benchmark(&plan, &secret, &fingerprint_target);
    let xpub = autotune_and_benchmark(&plan, &secret, &target);

    assert_eq!(xfp.matches, 1);
    assert_eq!(xpub.matches, 1);
    assert!(xfp.sustained_checks_per_second.is_finite());
    assert!(xpub.sustained_checks_per_second.is_finite());
    assert!(xfp.sustained_checks_per_second > 0.0);
    assert!(xpub.sustained_checks_per_second > 0.0);
}

fn autotune_and_benchmark(
    plan: &recoverme::search::RecoveryPlan,
    secret: &SecretMnemonic,
    target: &VerificationTarget,
) -> SustainedBenchmarkResult {
    let selected = TUNING_BATCH_SIZES
        .map(|batch_size| {
            let result = benchmark_sustained_pipeline(
                plan,
                secret,
                target,
                recoverme::domain::BackendKind::Cuda,
                benchmark_config(batch_size, TUNING_REPETITIONS),
            )
            .unwrap();
            print_result("tuning", &result);
            result
        })
        .into_iter()
        .max_by(|left, right| {
            left.sustained_checks_per_second
                .total_cmp(&right.sustained_checks_per_second)
        })
        .unwrap();

    let result = benchmark_sustained_pipeline(
        plan,
        secret,
        target,
        recoverme::domain::BackendKind::Cuda,
        benchmark_config(selected.batch_size, SUSTAINED_REPETITIONS),
    )
    .unwrap();
    print_result("sustained", &result);
    result
}

fn benchmark_config(batch_size: usize, repetitions: usize) -> SustainedBenchmarkConfig {
    SustainedBenchmarkConfig {
        sample_size: NonZeroUsize::new(batch_size).unwrap(),
        repetitions: NonZeroUsize::new(repetitions).unwrap(),
        batch_size: NonZeroUsize::new(batch_size).unwrap(),
        workgroup_size: NonZeroU32::new(WORKGROUP_SIZE),
        through: SearchPhase::WrittenCase,
    }
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
        "GPUQ_BENCHMARK stage={stage} mode={} backend={} device={} sample={} repetitions={} batch={} workgroup={} sustained_checks_per_second={:.1} median_checks_per_second={:.1} minimum_checks_per_second={:.1} maximum_checks_per_second={:.1} matches={} iteration_milliseconds={:?} warmup_excluded=true",
        result.verification_mode,
        result.backend,
        result.device,
        result.sample_size,
        result.repetitions,
        result.batch_size,
        result
            .workgroup_size
            .map_or_else(|| "n/a".into(), |size| size.to_string()),
        result.sustained_checks_per_second,
        result.median_checks_per_second,
        result.minimum_checks_per_second,
        result.maximum_checks_per_second,
        result.matches,
        result.iteration_milliseconds
    );
}
