use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::error::RecoverError;

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    pub mnemonic_file: Option<PathBuf>,
    pub template_file: Option<PathBuf>,
    pub passphrase_file: Option<PathBuf>,
    pub empty_passphrase: Option<bool>,
    pub words_file: Option<PathBuf>,
    pub recipe_file: Option<PathBuf>,
    pub fingerprint: Option<String>,
    pub master_xpub_file: Option<PathBuf>,
    pub state_dir: Option<PathBuf>,
    pub neighbors: Option<usize>,
    pub max_replacements: Option<usize>,
    pub lowercase_already_tried: Option<bool>,
    pub order: Option<String>,
    pub spacing: Option<String>,
    pub concatenated_already_tried: Option<bool>,
}

/// Load `--config` or `RECOVERME_CONFIG` before clap resolves environment defaults
pub fn apply_config_defaults() -> Result<(), RecoverError> {
    let Some(path) = config_path() else {
        return Ok(());
    };
    let config = load_file_config(&path)?;

    set_path("RECOVERME_MNEMONIC_FILE", config.mnemonic_file);
    set_path("RECOVERME_TEMPLATE_FILE", config.template_file);
    set_path("RECOVERME_PASSPHRASE_FILE", config.passphrase_file);
    if config.empty_passphrase == Some(true) {
        set_display("RECOVERME_EMPTY_PASSPHRASE", Some(true));
    }
    set_path("RECOVERME_WORDS_FILE", config.words_file);
    set_path("RECOVERME_RECIPE_FILE", config.recipe_file);
    set_value("RECOVERME_FINGERPRINT", config.fingerprint);
    set_path("RECOVERME_MASTER_XPUB_FILE", config.master_xpub_file);
    set_path("RECOVERME_STATE_DIR", config.state_dir);
    set_display("RECOVERME_NEIGHBORS", config.neighbors);
    set_display("RECOVERME_MAX_REPLACEMENTS", config.max_replacements);
    set_display(
        "RECOVERME_LOWERCASE_ALREADY_TRIED",
        config.lowercase_already_tried,
    );
    set_value("RECOVERME_ORDER", config.order);
    set_value("RECOVERME_SPACING", config.spacing);
    set_display(
        "RECOVERME_CONCATENATED_ALREADY_TRIED",
        config.concatenated_already_tried,
    );
    Ok(())
}

fn load_file_config(path: &Path) -> Result<FileConfig, RecoverError> {
    check_owner_only(path)?;
    let text = fs::read_to_string(path).map_err(|error| RecoverError::io(path, error))?;
    toml::from_str(&text)
        .map_err(|error| RecoverError::InvalidSetting(format!("invalid config file: {error}")))
}

fn config_path() -> Option<PathBuf> {
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--config" {
            return arguments.next().map(PathBuf::from);
        }
        if let Some(value) = argument.to_string_lossy().strip_prefix("--config=") {
            return Some(PathBuf::from(value));
        }
    }
    env::var_os("RECOVERME_CONFIG").map(PathBuf::from)
}

fn set_path(name: &'static str, value: Option<PathBuf>) {
    set_os(name, value.map(PathBuf::into_os_string));
}

fn set_value(name: &'static str, value: Option<String>) {
    set_os(name, value.map(OsString::from));
}

fn set_display(name: &'static str, value: Option<impl ToString>) {
    set_value(name, value.map(|value| value.to_string()));
}

fn set_os(name: &'static str, value: Option<OsString>) {
    if env::var_os(name).is_none() {
        if let Some(value) = value {
            env::set_var(name, value);
        }
    }
}

fn check_owner_only(path: &Path) -> Result<(), RecoverError> {
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
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn owner_only_config_parses_recovery_defaults() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("recoverme.toml");
        fs::write(
            &path,
            "state_dir = 'state'\nmax_replacements = 3\nspacing = 'coldcard'\n",
        )
        .unwrap();
        secure(&path);

        let config = load_file_config(&path).unwrap();

        assert_eq!(config.state_dir, Some(PathBuf::from("state")));
        assert_eq!(config.max_replacements, Some(3));
        assert_eq!(config.spacing.as_deref(), Some("coldcard"));
    }

    #[cfg(unix)]
    #[test]
    fn config_rejects_group_readable_files() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let path = directory.path().join("recoverme.toml");
        fs::write(&path, "state_dir = 'state'\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();

        assert!(matches!(
            load_file_config(&path),
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
