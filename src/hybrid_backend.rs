use crate::{
    crypto::{CpuSeedDeriver, RecoveryBackend, SeedBatch},
    cube_backend::CubeSeedDeriver,
    domain::{BackendKind, CandidateBatch, SecretMnemonic, VerificationTarget},
    error::RecoverError,
};

const CPU_SHARE_PERCENT: usize = 35;

/// Concurrent ARM CPU and Metal recovery backend for Apple silicon
pub struct HybridBackend {
    cpu: CpuSeedDeriver,
    metal: CubeSeedDeriver,
    batch_size: usize,
}

impl HybridBackend {
    /// Construct CPU and Metal workers sharing one mnemonic
    pub fn new(mnemonic: &SecretMnemonic) -> Result<Self, RecoverError> {
        Ok(Self {
            cpu: CpuSeedDeriver::new(mnemonic)?,
            metal: CubeSeedDeriver::metal(mnemonic)?,
            batch_size: 65_536,
        })
    }

    fn split(
        &self,
        candidates: &CandidateBatch,
    ) -> Result<(CandidateBatch, CandidateBatch), RecoverError> {
        let cpu_count = candidates.len() * CPU_SHARE_PERCENT / 100;
        let (cpu, metal) = candidates.candidates().split_at(cpu_count);
        Ok((
            CandidateBatch::new(cpu.to_vec())?,
            CandidateBatch::new(metal.to_vec())?,
        ))
    }
}

impl RecoveryBackend for HybridBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Hybrid
    }

    fn device_name(&self) -> String {
        format!("{} + {}", self.cpu.device_name(), self.metal.device_name())
    }

    fn preferred_batch_size(&self) -> usize {
        self.batch_size
    }

    fn workgroup_size(&self) -> Option<u32> {
        self.metal.workgroup_size()
    }

    fn configure(
        &mut self,
        batch_size: usize,
        workgroup_size: Option<u32>,
    ) -> Result<(), RecoverError> {
        if batch_size == 0 {
            return Err(RecoverError::InvalidSetting(
                "backend batch size must be greater than zero".into(),
            ));
        }
        self.batch_size = batch_size;
        self.metal.configure(batch_size, workgroup_size)
    }

    fn derive_seeds(&mut self, candidates: &CandidateBatch) -> Result<SeedBatch, RecoverError> {
        let (cpu_batch, metal_batch) = self.split(candidates)?;
        let (cpu, metal) = (&mut self.cpu, &mut self.metal);
        let (cpu_seeds, metal_seeds) = std::thread::scope(|scope| {
            let cpu_worker = scope.spawn(|| cpu.derive_seeds(&cpu_batch));
            let metal_seeds = metal.derive_seeds(&metal_batch)?;
            let cpu_seeds = cpu_worker
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)??;
            Ok::<_, RecoverError>((cpu_seeds, metal_seeds))
        })?;
        let mut seeds = Vec::with_capacity(candidates.len());
        seeds.extend_from_slice(cpu_seeds.as_slice());
        seeds.extend_from_slice(metal_seeds.as_slice());
        Ok(SeedBatch::new(seeds))
    }

    fn verify(
        &mut self,
        candidates: &CandidateBatch,
        target: &VerificationTarget,
    ) -> Result<Vec<usize>, RecoverError> {
        let (cpu_batch, metal_batch) = self.split(candidates)?;
        let metal_offset = cpu_batch.len();
        let (cpu, metal) = (&mut self.cpu, &mut self.metal);
        let (mut cpu_matches, metal_matches) = std::thread::scope(|scope| {
            let cpu_worker = scope.spawn(|| cpu.verify(&cpu_batch, target));
            let metal_matches = metal.verify(&metal_batch, target)?;
            let cpu_matches = cpu_worker
                .join()
                .map_err(|_| RecoverError::CandidatePreparationPanic)??;
            Ok::<_, RecoverError>((cpu_matches, metal_matches))
        })?;
        cpu_matches.extend(metal_matches.into_iter().map(|index| index + metal_offset));
        Ok(cpu_matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Candidate, CandidateId, SearchPhase};

    const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

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
        let mut hybrid = HybridBackend::new(&mnemonic).unwrap();
        let hybrid_seeds = hybrid.derive_seeds(&candidates).unwrap();
        let mut cpu = CpuSeedDeriver::new(&mnemonic).unwrap();
        let cpu_seeds = cpu.derive_seeds(&candidates).unwrap();

        assert_eq!(hybrid_seeds.as_slice(), cpu_seeds.as_slice());
    }
}
