use crate::{
    crypto::{CpuSeedDeriver, RecoveryBackend, SeedBatch},
    cube_backend::CubeSeedDeriver,
    domain::{
        BackendConfiguration, BackendKind, CandidateBatch, CpuShare, SecretMnemonic,
        VerificationTarget,
    },
    error::RecoverError,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
const METAL_CPU_SHARE_PERCENT: u8 = 35;

/// Concurrent CPU and Metal recovery backend
pub struct HybridBackend {
    cpu: CpuSeedDeriver,
    accelerator: CubeSeedDeriver,
    cpu_share: CpuShare,
    batch_size: usize,
}

impl HybridBackend {
    /// Construct the fixed-share Apple silicon CPU and Metal backend
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal(mnemonic: &SecretMnemonic) -> Result<Self, RecoverError> {
        Ok(Self {
            cpu: CpuSeedDeriver::new(mnemonic)?,
            accelerator: CubeSeedDeriver::metal(mnemonic)?,
            cpu_share: METAL_CPU_SHARE_PERCENT.try_into()?,
            batch_size: 65_536,
        })
    }

    fn split(
        &self,
        candidates: &CandidateBatch,
    ) -> Result<(CandidateBatch, CandidateBatch), RecoverError> {
        split_at_cpu_share(candidates, self.cpu_share)
    }
}

fn split_at_cpu_share(
    candidates: &CandidateBatch,
    cpu_share: CpuShare,
) -> Result<(CandidateBatch, CandidateBatch), RecoverError> {
    let cpu_count = candidates.len() * usize::from(cpu_share.percent()) / 100;
    let (cpu, accelerator) = candidates.candidates().split_at(cpu_count);
    Ok((
        CandidateBatch::new(cpu.to_vec())?,
        CandidateBatch::new(accelerator.to_vec())?,
    ))
}

impl RecoveryBackend for HybridBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Hybrid
    }

    fn device_name(&self) -> String {
        format!(
            "{} + {}",
            self.cpu.device_name(),
            self.accelerator.device_name()
        )
    }

    fn preferred_batch_size(&self) -> usize {
        self.batch_size
    }

    fn workgroup_size(&self) -> Option<u32> {
        self.accelerator.workgroup_size()
    }

    fn configure(&mut self, configuration: BackendConfiguration) -> Result<(), RecoverError> {
        let BackendConfiguration::Hybrid {
            batch_size,
            workgroup_size,
            cpu_share,
        } = configuration
        else {
            return Err(RecoverError::InvalidSetting(
                "hybrid backend requires a hybrid configuration".into(),
            ));
        };
        self.batch_size = batch_size.get();
        self.cpu_share = cpu_share;
        self.accelerator.configure(BackendConfiguration::Cube {
            batch_size,
            workgroup_size,
        })
    }

    fn cpu_share_percent(&self) -> Option<u8> {
        Some(self.cpu_share.percent())
    }

    fn derive_seeds(&mut self, candidates: &CandidateBatch) -> Result<SeedBatch, RecoverError> {
        let (cpu_batch, accelerator_batch) = self.split(candidates)?;
        let (cpu, accelerator) = (&mut self.cpu, &mut self.accelerator);
        let (cpu_seeds, accelerator_seeds) = std::thread::scope(|scope| {
            let cpu_worker = scope.spawn(|| cpu.derive_seeds(&cpu_batch));
            let accelerator_seeds = accelerator.derive_seeds(&accelerator_batch)?;
            let cpu_seeds = cpu_worker
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)??;
            Ok::<_, RecoverError>((cpu_seeds, accelerator_seeds))
        })?;
        let mut seeds = Vec::with_capacity(candidates.len());
        seeds.extend_from_slice(cpu_seeds.as_slice());
        seeds.extend_from_slice(accelerator_seeds.as_slice());
        Ok(SeedBatch::new(seeds))
    }

    fn verify(
        &mut self,
        candidates: &CandidateBatch,
        target: &VerificationTarget,
    ) -> Result<Vec<usize>, RecoverError> {
        let (cpu_batch, accelerator_batch) = self.split(candidates)?;
        let accelerator_offset = cpu_batch.len();
        let (cpu, accelerator) = (&mut self.cpu, &mut self.accelerator);
        let (mut cpu_matches, accelerator_matches) = std::thread::scope(|scope| {
            let cpu_worker = scope.spawn(|| cpu.verify(&cpu_batch, target));
            let accelerator_matches = accelerator.verify(&accelerator_batch, target)?;
            let cpu_matches = cpu_worker
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)??;
            Ok::<_, RecoverError>((cpu_matches, accelerator_matches))
        })?;
        cpu_matches.extend(
            accelerator_matches
                .into_iter()
                .map(|index| index + accelerator_offset),
        );
        Ok(cpu_matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Candidate, CandidateId, SearchPhase};

    #[cfg(all(feature = "metal", target_os = "macos"))]
    const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn preserves_candidate_order_across_workers() {
        let mnemonic = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let candidates = CandidateBatch::new(
            (0..8)
                .map(|index| {
                    Candidate::new(
                        CandidateId(index.to_string()),
                        SearchPhase::WrittenLower,
                        vec![format!("word{index}")],
                    )
                })
                .collect(),
        )
        .unwrap();
        let mut hybrid = HybridBackend::metal(&mnemonic).unwrap();
        let hybrid_seeds = hybrid.derive_seeds(&candidates).unwrap();
        let mut cpu = CpuSeedDeriver::new(&mnemonic).unwrap();
        let cpu_seeds = cpu.derive_seeds(&candidates).unwrap();

        assert_eq!(hybrid_seeds.as_slice(), cpu_seeds.as_slice());
    }

    #[test]
    fn split_preserves_every_candidate_at_tuned_share() {
        let candidates = CandidateBatch::new(
            (0..101)
                .map(|index| {
                    Candidate::new(
                        CandidateId(index.to_string()),
                        SearchPhase::WrittenLower,
                        vec![format!("word{index}")],
                    )
                })
                .collect(),
        )
        .unwrap();

        let (cpu, accelerator) =
            split_at_cpu_share(&candidates, CpuShare::try_from(2).unwrap()).unwrap();

        assert_eq!(cpu.len(), 2);
        assert_eq!(accelerator.len(), 99);
        assert_eq!(cpu.candidates()[0].passphrase(), "word0");
        assert_eq!(accelerator.candidates()[0].passphrase(), "word2");
        assert_eq!(accelerator.candidates()[98].passphrase(), "word100");
    }
}
