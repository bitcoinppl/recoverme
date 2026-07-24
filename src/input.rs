use std::{env, fs, path::Path};

use bip39::{Language, Mnemonic};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

const SEED_ENVIRONMENT_VARIABLE: &str = "RECOVERME_MNEMONIC";
const PASSPHRASE_ENVIRONMENT_VARIABLE: &str = "RECOVERME_WORDS";
const XFP_ENVIRONMENT_VARIABLE: &str = "RECOVERME_FINGERPRINT";

use crate::{
    domain::{
        MasterXpubTarget, RecoveryRecipe, RecoverySettings, SecretMnemonic, TargetFingerprint,
        TokenSlot, VerificationTarget, WrittenWords,
    },
    error::RecoverError,
};

/// Validated secret inputs used to compile and execute a recovery plan
#[derive(Debug, Clone)]
pub struct RecoveryInputs {
    /// Secret BIP39 mnemonic
    pub mnemonic: SecretMnemonic,
    /// Normalized passphrase words in written order
    pub written_words: WrittenWords,
    /// Typed recipe used by the candidate planner
    pub recipe: RecoveryRecipe,
}

/// Read and validate the protected mnemonic and written-word files
pub fn load_inputs(
    mnemonic_path: &Path,
    words_path: &Path,
) -> Result<RecoveryInputs, RecoverError> {
    check_secret_file(mnemonic_path)?;
    check_secret_file(words_path)?;

    let mnemonic_text = Zeroizing::new(read_secret(mnemonic_path)?);
    let words_text = Zeroizing::new(read_secret(words_path)?);
    let words = words_text.lines().enumerate().filter_map(|(index, line)| {
        let word = line.trim();
        (!word.is_empty()).then_some((index + 1, word))
    });

    parse_inputs(&mnemonic_text, words)
}

/// Read and validate the scoped mnemonic and whitespace-separated words variables
pub fn load_inputs_from_env() -> Result<RecoveryInputs, RecoverError> {
    let mnemonic_text = read_environment(SEED_ENVIRONMENT_VARIABLE)?;
    let words_text = read_environment(PASSPHRASE_ENVIRONMENT_VARIABLE)?;
    let words = words_text
        .split_whitespace()
        .enumerate()
        .map(|(index, word)| (index + 1, word));
    parse_inputs(&mnemonic_text, words)
}

/// Read and validate the scoped target fingerprint variable
pub fn load_target_fingerprint_from_env() -> Result<TargetFingerprint, RecoverError> {
    read_environment(XFP_ENVIRONMENT_VARIABLE)?.parse()
}

fn parse_inputs<'a>(
    mnemonic_text: &str,
    words: impl Iterator<Item = (usize, &'a str)>,
) -> Result<RecoveryInputs, RecoverError> {
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_text.trim())
        .map_err(|error| RecoverError::InvalidMnemonic(error.to_string()))?;
    let words = words
        .map(|(line, word)| normalize_written_word(line, word))
        .collect::<Result<Vec<_>, _>>()?;

    let written_words = WrittenWords::new(words)?;
    let recipe = RecoveryRecipe::from_written_words(&written_words);
    Ok(RecoveryInputs {
        mnemonic: SecretMnemonic::new(mnemonic.to_string()),
        written_words,
        recipe,
    })
}

#[derive(serde::Deserialize)]
struct RecipeDocument {
    version: u32,
    slots: Vec<RecipeSlotDocument>,
}

#[derive(serde::Deserialize)]
struct RecipeSlotDocument {
    alternatives: Vec<String>,
    #[serde(default)]
    optional: bool,
}

/// Read an owner-only advanced recipe file
pub fn load_recipe(path: &Path) -> Result<RecoveryRecipe, RecoverError> {
    check_secret_file(path)?;
    let text = Zeroizing::new(read_secret(path)?);
    let document: RecipeDocument = toml::from_str(&text)
        .map_err(|error| RecoverError::InvalidSetting(format!("invalid recipe file: {error}")))?;
    if document.version != 1 {
        return Err(RecoverError::InvalidSetting(format!(
            "unsupported recipe version {}",
            document.version
        )));
    }
    RecoveryRecipe::new(
        document
            .slots
            .into_iter()
            .map(|slot| TokenSlot::new(slot.alternatives, slot.optional))
            .collect::<Result<Vec<_>, _>>()?,
    )
}

/// Read a mnemonic and advanced recipe from protected files
pub fn load_inputs_with_recipe(
    mnemonic_path: &Path,
    recipe_path: &Path,
) -> Result<RecoveryInputs, RecoverError> {
    check_secret_file(mnemonic_path)?;
    let mnemonic_text = Zeroizing::new(read_secret(mnemonic_path)?);
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_text.trim())
        .map_err(|error| RecoverError::InvalidMnemonic(error.to_string()))?;
    let recipe = load_recipe(recipe_path)?;
    let primary_words = recipe
        .slots()
        .iter()
        .map(|slot| slot.alternatives()[0].clone())
        .collect();
    Ok(RecoveryInputs {
        mnemonic: SecretMnemonic::new(mnemonic.to_string()),
        written_words: WrittenWords::new(primary_words)?,
        recipe,
    })
}

/// Compute the secret-independent identifier stored in a recovery manifest
pub fn recovery_spec_hash(
    inputs: &RecoveryInputs,
    fingerprint: TargetFingerprint,
    settings: &RecoverySettings,
) -> String {
    recovery_spec_hash_for_target(
        inputs,
        &VerificationTarget::Fingerprint(fingerprint),
        settings,
    )
}

/// Read and validate a protected depth-zero master extended public key
pub fn load_master_xpub(path: &Path) -> Result<MasterXpubTarget, RecoverError> {
    check_secret_file(path)?;
    MasterXpubTarget::parse(&read_secret(path)?)
}

/// Compute the identifier stored for secret inputs, target, and immutable settings
pub fn recovery_spec_hash_for_target(
    inputs: &RecoveryInputs,
    target: &VerificationTarget,
    settings: &RecoverySettings,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"recoverme-spec-v4\0");
    digest.update(inputs.mnemonic.expose().as_bytes());
    digest.update([0]);
    for slot in inputs.recipe.slots() {
        digest.update([u8::from(slot.is_optional())]);
        for alternative in slot.alternatives() {
            digest.update(alternative.as_bytes());
            digest.update([0]);
        }
        digest.update([0xff]);
    }
    digest.update(target.fingerprint().bytes());
    if let Some(master_xpub) = target.master_xpub() {
        digest.update([1]);
        digest.update(master_xpub.public_key());
        digest.update(master_xpub.chain_code());
    } else {
        digest.update([0]);
    }
    digest.update(settings.neighbors_per_word.to_le_bytes());
    digest.update(settings.max_replacements.to_le_bytes());
    digest.update(settings.local_swap_radius.to_le_bytes());
    digest.update(settings.max_passphrase_bytes.to_le_bytes());
    digest.update([u8::from(settings.lowercase_already_tried)]);
    digest.update([settings.order as u8]);
    digest.update([settings.spacing as u8]);
    digest.update([u8::from(settings.concatenated_already_tried)]);
    hex::encode(digest.finalize())
}

fn normalize_written_word(line: usize, value: &str) -> Result<String, RecoverError> {
    if !value.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        return Err(RecoverError::InvalidWrittenWord { line });
    }
    Ok(value.to_ascii_lowercase())
}

fn read_environment(name: &'static str) -> Result<Zeroizing<String>, RecoverError> {
    match env::var(name) {
        Ok(value) => Ok(Zeroizing::new(value)),
        Err(env::VarError::NotPresent) => Err(RecoverError::MissingEnvironmentVariable(name)),
        Err(env::VarError::NotUnicode(_)) => Err(RecoverError::InvalidEnvironmentVariable(name)),
    }
}

pub(crate) fn read_secret(path: &Path) -> Result<String, RecoverError> {
    fs::read_to_string(path).map_err(|error| RecoverError::io(path, error))
}

pub(crate) fn check_secret_file(path: &Path) -> Result<(), RecoverError> {
    let metadata = fs::metadata(path).map_err(|error| RecoverError::io(path, error))?;
    if !metadata.is_file() {
        return Err(RecoverError::io(
            path,
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a regular file"),
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(RecoverError::InsecurePermissions(path.to_owned()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, fs, sync::Mutex};

    use tempfile::tempdir;

    use super::*;

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());

    struct EnvironmentRestore(Vec<(&'static str, Option<OsString>)>);

    impl EnvironmentRestore {
        fn set(values: &[(&'static str, &str)]) -> Self {
            let previous = values
                .iter()
                .map(|(name, _)| (*name, env::var_os(name)))
                .collect();
            for (name, value) in values {
                env::set_var(name, value);
            }
            Self(previous)
        }
    }

    impl Drop for EnvironmentRestore {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                if let Some(value) = value {
                    env::set_var(name, value);
                } else {
                    env::remove_var(name);
                }
            }
        }
    }

    #[test]
    fn written_words_are_normalized_without_joining_them() {
        let directory = tempdir().unwrap();
        let mnemonic_path = directory.path().join("mnemonic");
        let words_path = directory.path().join("words");
        fs::write(&mnemonic_path, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art\n").unwrap();
        fs::write(&words_path, "Alpha\nBRISK\n").unwrap();
        secure(&mnemonic_path);
        secure(&words_path);

        let inputs = load_inputs(&mnemonic_path, &words_path).unwrap();

        assert_eq!(
            inputs.written_words.as_slice(),
            &["alpha".to_owned(), "brisk".to_owned()]
        );
    }

    #[test]
    fn environment_inputs_use_whitespace_separated_written_words() {
        let _lock = ENVIRONMENT_LOCK.lock().unwrap();
        let _restore = EnvironmentRestore::set(&[
            (
                SEED_ENVIRONMENT_VARIABLE,
                "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art",
            ),
            (PASSPHRASE_ENVIRONMENT_VARIABLE, "Alpha  BRISK\ncactus"),
            (XFP_ENVIRONMENT_VARIABLE, "12345678"),
        ]);

        let inputs = load_inputs_from_env().unwrap();
        let fingerprint = load_target_fingerprint_from_env().unwrap();

        assert_eq!(
            inputs.written_words.as_slice(),
            &["alpha".to_owned(), "brisk".to_owned(), "cactus".to_owned()]
        );
        assert_eq!(fingerprint.to_string(), "12345678");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_group_readable_secret_files() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let mnemonic_path = directory.path().join("mnemonic");
        let words_path = directory.path().join("words");
        fs::write(&mnemonic_path, "invalid").unwrap();
        fs::write(&words_path, "word").unwrap();
        fs::set_permissions(&mnemonic_path, fs::Permissions::from_mode(0o640)).unwrap();
        secure(&words_path);

        assert!(matches!(
            load_inputs(&mnemonic_path, &words_path),
            Err(RecoverError::InsecurePermissions(path)) if path == mnemonic_path
        ));
    }

    #[cfg(unix)]
    #[test]
    fn master_xpub_file_must_be_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let path = directory.path().join("master-xpub");
        fs::write(&path, "not-an-xpub\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        assert!(matches!(
            load_master_xpub(&path),
            Err(RecoverError::InsecurePermissions(insecure)) if insecure == path
        ));
    }

    fn secure(path: &Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
        }
    }
}
