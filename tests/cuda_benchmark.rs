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
const SAMPLE_SIZE: usize = 65_536;
const REPETITIONS: usize = 5;
const BATCH_SIZE: usize = 65_536;
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
    let config = SustainedBenchmarkConfig {
        sample_size: NonZeroUsize::new(SAMPLE_SIZE).unwrap(),
        repetitions: NonZeroUsize::new(REPETITIONS).unwrap(),
        batch_size: NonZeroUsize::new(BATCH_SIZE).unwrap(),
        workgroup_size: NonZeroU32::new(WORKGROUP_SIZE),
        through: SearchPhase::WrittenCase,
    };

    let xfp = benchmark_sustained_pipeline(
        &plan,
        &secret,
        &fingerprint_target,
        recoverme::domain::BackendKind::Cuda,
        config,
    )
    .unwrap();
    print_result(&xfp);

    let xpub = benchmark_sustained_pipeline(
        &plan,
        &secret,
        &target,
        recoverme::domain::BackendKind::Cuda,
        config,
    )
    .unwrap();
    print_result(&xpub);

    assert_eq!(xfp.matches, 1);
    assert_eq!(xpub.matches, 1);
    assert!(xfp.sustained_checks_per_second.is_finite());
    assert!(xpub.sustained_checks_per_second.is_finite());
    assert!(xfp.sustained_checks_per_second > 0.0);
    assert!(xpub.sustained_checks_per_second > 0.0);
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

fn print_result(result: &SustainedBenchmarkResult) {
    println!(
        "GPUQ_BENCHMARK mode={} backend={} device={} sample={} repetitions={} batch={} workgroup={} sustained_checks_per_second={:.1} median_checks_per_second={:.1} minimum_checks_per_second={:.1} maximum_checks_per_second={:.1} matches={} iteration_milliseconds={:?} warmup_excluded=true",
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
