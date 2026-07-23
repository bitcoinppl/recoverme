use crate::{
    crypto::{CpuSeedDeriver, RecoveryBackend},
    domain::{BackendKind, SecretMnemonic},
    error::RecoverError,
    state::RecoveryState,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::hybrid_backend::HybridBackend;

#[cfg(any(
    feature = "cube-cpu",
    all(feature = "metal", target_os = "macos"),
    feature = "cuda"
))]
use crate::cube_backend::CubeSeedDeriver;

/// Backends compiled into this executable on the current platform
pub fn available_backends() -> Vec<BackendKind> {
    let mut backends = vec![BackendKind::Cpu];
    backends.extend(compiled_cube_backends());
    backends
}

fn compiled_cube_backends() -> Vec<BackendKind> {
    let backends: &[BackendKind] = &[
        #[cfg(feature = "cube-cpu")]
        BackendKind::CubeCpu,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BackendKind::Metal,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BackendKind::Hybrid,
        #[cfg(feature = "cuda")]
        BackendKind::Cuda,
    ];
    backends.to_vec()
}

/// Resolve `auto` to the fastest benchmarked available backend
pub fn resolve_backend(requested: BackendKind, state: &RecoveryState) -> BackendKind {
    if requested != BackendKind::Auto {
        return requested;
    }

    available_backends()
        .into_iter()
        .filter_map(|backend| {
            state
                .latest_benchmark(backend)
                .map(|benchmark| (backend, benchmark.checks_per_second))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map_or(BackendKind::Cpu, |(backend, _)| backend)
}

/// Construct a seed deriver for a concrete backend
pub fn create_deriver(
    backend: BackendKind,
    mnemonic: &SecretMnemonic,
) -> Result<Box<dyn RecoveryBackend>, RecoverError> {
    match backend {
        BackendKind::Auto => Err(RecoverError::BackendUnavailable(
            "auto must be resolved before backend construction".into(),
        )),
        BackendKind::Cpu => Ok(Box::new(CpuSeedDeriver::new(mnemonic)?)),
        #[cfg(feature = "cube-cpu")]
        BackendKind::CubeCpu => Ok(Box::new(CubeSeedDeriver::cpu(mnemonic)?)),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BackendKind::Metal => Ok(Box::new(CubeSeedDeriver::metal(mnemonic)?)),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BackendKind::Hybrid => Ok(Box::new(HybridBackend::new(mnemonic)?)),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => Ok(Box::new(CubeSeedDeriver::cuda(mnemonic)?)),
        unavailable => Err(RecoverError::BackendUnavailable(unavailable.to_string())),
    }
}
