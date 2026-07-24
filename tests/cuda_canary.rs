use bip32::{Prefix, XPrv};
use bip39::{Language, Mnemonic};
use recoverme::{
    crypto::RecoveryBackend,
    cube_backend::CubeSeedDeriver,
    domain::{Candidate, CandidateId},
    CandidateBatch, MasterXpubTarget, SearchPhase, SecretMnemonic, TargetFingerprint,
    VerificationTarget,
};

const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
const PUBLIC_TEST_PASSPHRASE: &str = "BenefitWIFE";

#[test]
#[ignore = "requires a CUDA device"]
fn cuda_public_fixture_matches_reference() {
    let secret = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
    let candidates = CandidateBatch::new(vec![
        Candidate::new(
            CandidateId("empty".into()),
            SearchPhase::WrittenLower,
            vec![String::new()],
        ),
        Candidate::new(
            CandidateId("wrong".into()),
            SearchPhase::WrittenLower,
            vec!["wrong".into()],
        ),
        Candidate::new(
            CandidateId("match".into()),
            SearchPhase::WrittenCase,
            vec!["Benefit".into(), "WIFE".into()],
        ),
    ])
    .unwrap();
    let mnemonic = Mnemonic::parse_in(Language::English, PUBLIC_TEST_MNEMONIC).unwrap();
    let expected = XPrv::new(mnemonic.to_seed(PUBLIC_TEST_PASSPHRASE)).unwrap();
    let master_xpub =
        MasterXpubTarget::parse(&expected.public_key().to_string(Prefix::XPUB)).unwrap();
    let fingerprint = hex::encode(master_xpub.fingerprint())
        .parse::<TargetFingerprint>()
        .unwrap();
    let target = VerificationTarget::new(fingerprint, Some(master_xpub)).unwrap();
    let mut deriver = CubeSeedDeriver::cuda(&secret).unwrap();

    let actual = deriver.derive_seeds(&candidates).unwrap();
    assert_eq!(actual.as_slice()[0], mnemonic.to_seed(""));
    assert_eq!(
        actual.as_slice()[2],
        mnemonic.to_seed(PUBLIC_TEST_PASSPHRASE)
    );
    assert_eq!(deriver.verify(&candidates, &target).unwrap(), [2]);
}
