use crate::{
    crypto::{CpuSeedDeriver, RecoveryBackend},
    domain::{BackendConfiguration, BackendKind, SecretMnemonic},
    error::RecoverError,
    state::RecoveryState,
};

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
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
        #[cfg(feature = "cuda")]
        BackendKind::CudaHybrid,
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
        BackendKind::Hybrid => Ok(Box::new(HybridBackend::metal(mnemonic)?)),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => Ok(Box::new(CubeSeedDeriver::cuda(mnemonic)?)),
        BackendKind::CudaHybrid => Err(RecoverError::BackendUnavailable(
            "cuda-hybrid requires a persisted autotuned configuration".into(),
        )),
        unavailable => Err(RecoverError::BackendUnavailable(unavailable.to_string())),
    }
}

/// Construct a concrete backend with a validated runtime configuration
pub fn create_configured_deriver(
    backend: BackendKind,
    mnemonic: &SecretMnemonic,
    configuration: BackendConfiguration,
) -> Result<Box<dyn RecoveryBackend>, RecoverError> {
    #[cfg(feature = "cuda")]
    if backend == BackendKind::CudaHybrid {
        let Some(cpu_share) = configuration.cpu_share() else {
            return Err(RecoverError::InvalidSetting(
                "cuda-hybrid requires a nonzero CPU share".into(),
            ));
        };
        let mut deriver = HybridBackend::cuda(mnemonic, cpu_share)?;
        deriver.configure(configuration)?;
        return Ok(Box::new(deriver));
    }

    let mut deriver = create_deriver(backend, mnemonic)?;
    deriver.configure(configuration)?;
    Ok(deriver)
}
