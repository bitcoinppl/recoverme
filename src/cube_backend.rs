use cubecl::prelude::*;
use sha2::{Digest, Sha512};
use zeroize::Zeroizing;

use crate::{
    crypto::{matching_candidate_indices_for_target, RecoveryBackend, SeedBatch},
    domain::{BackendKind, CandidateBatch, SecretMnemonic, VerificationTarget},
    error::RecoverError,
};

const SHA512_INITIAL: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];
const CUBE_WORKGROUP_SIZE: u32 = 64;
const CUBE_BATCH_SIZE: usize = 65_536;

const SHA512_CONSTANTS: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

#[derive(Debug, Clone, Copy)]
enum CubeRuntimeKind {
    #[cfg(feature = "cube-cpu")]
    Cpu,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
}

/// CubeCL implementation of BIP39 PBKDF2-HMAC-SHA512
pub struct CubeSeedDeriver {
    kind: CubeRuntimeKind,
    inner_state: Zeroizing<Vec<u64>>,
    outer_state: Zeroizing<Vec<u64>>,
    inner_handle: cubecl::server::Handle,
    outer_handle: cubecl::server::Handle,
    bip32_inner_state: [u64; 8],
    bip32_outer_state: [u64; 8],
    bip32_inner_handle: cubecl::server::Handle,
    bip32_outer_handle: cubecl::server::Handle,
    batch_size: usize,
    workgroup_size: u32,
    device_name: String,
}

impl CubeSeedDeriver {
    /// Construct the CubeCL CPU runtime
    #[cfg(feature = "cube-cpu")]
    pub fn cpu(mnemonic: &SecretMnemonic) -> Result<Self, RecoverError> {
        Self::new::<cubecl::cpu::CpuRuntime>(CubeRuntimeKind::Cpu, mnemonic)
    }

    /// Construct the CubeCL Metal runtime backed by wgpu MSL
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal(mnemonic: &SecretMnemonic) -> Result<Self, RecoverError> {
        Self::new::<cubecl::wgpu::WgpuRuntime>(CubeRuntimeKind::Metal, mnemonic)
    }

    /// Construct the CubeCL CUDA runtime
    #[cfg(feature = "cuda")]
    pub fn cuda(mnemonic: &SecretMnemonic) -> Result<Self, RecoverError> {
        Self::new::<cubecl::cuda::CudaRuntime>(CubeRuntimeKind::Cuda, mnemonic)
    }

    fn new<R>(kind: CubeRuntimeKind, mnemonic: &SecretMnemonic) -> Result<Self, RecoverError>
    where
        R: Runtime,
        R::Device: Default,
    {
        let (inner_state, outer_state) = prepared_hmac_states(mnemonic.expose());
        let device = R::Device::default();
        let client = R::client(&device);
        let inner_handle = client.create_from_slice(u64::as_bytes(&inner_state));
        let outer_handle = client.create_from_slice(u64::as_bytes(&outer_state));
        let (bip32_inner_state, bip32_outer_state) = prepared_hmac_states("Bitcoin seed");
        let bip32_inner_handle = client.create_from_slice(u64::as_bytes(&bip32_inner_state));
        let bip32_outer_handle = client.create_from_slice(u64::as_bytes(&bip32_outer_state));
        Ok(Self {
            kind,
            inner_state: Zeroizing::new(inner_state.to_vec()),
            outer_state: Zeroizing::new(outer_state.to_vec()),
            inner_handle,
            outer_handle,
            bip32_inner_state,
            bip32_outer_state,
            bip32_inner_handle,
            bip32_outer_handle,
            batch_size: CUBE_BATCH_SIZE,
            workgroup_size: CUBE_WORKGROUP_SIZE,
            device_name: format!("{:?}", R::name(&client)),
        })
    }
}

impl RecoveryBackend for CubeSeedDeriver {
    fn kind(&self) -> BackendKind {
        match self.kind {
            #[cfg(feature = "cube-cpu")]
            CubeRuntimeKind::Cpu => BackendKind::CubeCpu,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            CubeRuntimeKind::Metal => BackendKind::Metal,
            #[cfg(feature = "cuda")]
            CubeRuntimeKind::Cuda => BackendKind::Cuda,
        }
    }

    fn device_name(&self) -> String {
        self.device_name.clone()
    }

    fn preferred_batch_size(&self) -> usize {
        self.batch_size
    }

    fn workgroup_size(&self) -> Option<u32> {
        Some(self.workgroup_size)
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
        let workgroup_size = workgroup_size.unwrap_or(self.workgroup_size);
        if !(32..=256).contains(&workgroup_size) || !workgroup_size.is_power_of_two() {
            return Err(RecoverError::InvalidSetting(
                "CubeCL workgroup size must be a power of two from 32 through 256".into(),
            ));
        }
        self.batch_size = batch_size;
        self.workgroup_size = workgroup_size;
        Ok(())
    }

    fn derive_seeds(&mut self, candidates: &CandidateBatch) -> Result<SeedBatch, RecoverError> {
        match self.kind {
            #[cfg(feature = "cube-cpu")]
            CubeRuntimeKind::Cpu => derive_with_runtime::<cubecl::cpu::CpuRuntime>(
                candidates,
                &self.inner_state,
                &self.outer_state,
                &self.inner_handle,
                &self.outer_handle,
                self.workgroup_size,
            ),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            CubeRuntimeKind::Metal => derive_with_runtime::<cubecl::wgpu::WgpuRuntime>(
                candidates,
                &self.inner_state,
                &self.outer_state,
                &self.inner_handle,
                &self.outer_handle,
                self.workgroup_size,
            ),
            #[cfg(feature = "cuda")]
            CubeRuntimeKind::Cuda => derive_with_runtime::<cubecl::cuda::CudaRuntime>(
                candidates,
                &self.inner_state,
                &self.outer_state,
                &self.inner_handle,
                &self.outer_handle,
                self.workgroup_size,
            ),
        }
    }

    fn verify(
        &mut self,
        candidates: &CandidateBatch,
        target: &VerificationTarget,
    ) -> Result<Vec<usize>, RecoverError> {
        let Some(master_xpub) = target.master_xpub() else {
            let seeds = self.derive_seeds(candidates)?;
            return matching_candidate_indices_for_target(&seeds, target);
        };
        let chain_code = master_xpub.chain_code();
        let possible = match self.kind {
            #[cfg(feature = "cube-cpu")]
            CubeRuntimeKind::Cpu => filter_chain_code_with_runtime::<cubecl::cpu::CpuRuntime>(
                candidates,
                self,
                chain_code,
                self.workgroup_size,
            )?,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            CubeRuntimeKind::Metal => filter_chain_code_with_runtime::<cubecl::wgpu::WgpuRuntime>(
                candidates,
                self,
                chain_code,
                self.workgroup_size,
            )?,
            #[cfg(feature = "cuda")]
            CubeRuntimeKind::Cuda => filter_chain_code_with_runtime::<cubecl::cuda::CudaRuntime>(
                candidates,
                self,
                chain_code,
                self.workgroup_size,
            )?,
        };
        if possible.is_empty() {
            return Ok(Vec::new());
        }

        let confirmation = CandidateBatch::new(
            possible
                .iter()
                .map(|index| candidates.candidates()[*index].clone())
                .collect(),
        )?;
        let seeds = self.derive_seeds(&confirmation)?;
        let confirmed = matching_candidate_indices_for_target(&seeds, target)?;
        Ok(confirmed.into_iter().map(|index| possible[index]).collect())
    }
}

fn derive_with_runtime<R: Runtime>(
    candidates: &CandidateBatch,
    inner_state: &[u64],
    outer_state: &[u64],
    inner_handle: &cubecl::server::Handle,
    outer_handle: &cubecl::server::Handle,
    workgroup_size: u32,
) -> Result<SeedBatch, RecoverError>
where
    R::Device: Default,
{
    if candidates.is_empty() {
        return Ok(SeedBatch::new(Vec::new()));
    }

    let stride = candidates.stride();
    let candidate_bytes = candidates.bytes();
    let lengths = candidates
        .lengths()
        .iter()
        .map(|length| u32::from(*length))
        .collect::<Vec<_>>();

    let device = R::Device::default();
    let client = R::client(&device);
    let candidates_handle = client.create_from_slice(u8::as_bytes(candidate_bytes));
    let lengths_handle = client.create_from_slice(u32::as_bytes(&lengths));
    debug_assert_eq!(inner_state.len(), 8);
    debug_assert_eq!(outer_state.len(), 8);
    let output_len = candidates.len() * 8;
    let output_handle = client.empty(output_len * core::mem::size_of::<u64>());
    let cube_dim = CubeDim::new_1d(workgroup_size);
    let cube_count = CubeCount::Static(
        candidates.len().div_ceil(workgroup_size as usize) as u32,
        1,
        1,
    );

    unsafe {
        bip39_seed_kernel::launch::<R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(candidates_handle, candidate_bytes.len()),
            ArrayArg::from_raw_parts(lengths_handle, lengths.len()),
            ArrayArg::from_raw_parts(inner_handle.clone(), inner_state.len()),
            ArrayArg::from_raw_parts(outer_handle.clone(), outer_state.len()),
            ArrayArg::from_raw_parts(output_handle.clone(), output_len),
            stride,
        );
    }

    let output_bytes = client
        .read_one(output_handle)
        .map_err(|error| RecoverError::SeedDerivation(format!("CubeCL readback: {error:?}")))?;
    let words = u64::from_bytes(&output_bytes);
    if words.len() != output_len {
        return Err(RecoverError::SeedDerivation(format!(
            "CubeCL returned {} words, expected {output_len}",
            words.len()
        )));
    }
    let seeds = words
        .chunks_exact(8)
        .map(|words| {
            let mut seed = [0_u8; 64];
            for (index, word) in words.iter().enumerate() {
                seed[index * 8..index * 8 + 8].copy_from_slice(&word.to_be_bytes());
            }
            seed
        })
        .collect();
    Ok(SeedBatch::new(seeds))
}

fn filter_chain_code_with_runtime<R: Runtime>(
    candidates: &CandidateBatch,
    deriver: &CubeSeedDeriver,
    chain_code: [u8; 32],
    workgroup_size: u32,
) -> Result<Vec<usize>, RecoverError>
where
    R::Device: Default,
{
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let lengths = candidates
        .lengths()
        .iter()
        .map(|length| u32::from(*length))
        .collect::<Vec<_>>();
    let expected = chain_code
        .chunks_exact(8)
        .map(|bytes| u64::from_be_bytes(bytes.try_into().expect("eight-byte chunk")))
        .collect::<Vec<_>>();
    let device = R::Device::default();
    let client = R::client(&device);
    let candidates_handle = client.create_from_slice(u8::as_bytes(candidates.bytes()));
    let lengths_handle = client.create_from_slice(u32::as_bytes(&lengths));
    let expected_handle = client.create_from_slice(u64::as_bytes(&expected));
    let output_handle = client.empty(candidates.len() * core::mem::size_of::<u32>());
    let cube_count = CubeCount::Static(
        candidates.len().div_ceil(workgroup_size as usize) as u32,
        1,
        1,
    );

    unsafe {
        bip39_chain_code_kernel::launch::<R>(
            &client,
            cube_count,
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts(candidates_handle, candidates.bytes().len()),
            ArrayArg::from_raw_parts(lengths_handle, lengths.len()),
            ArrayArg::from_raw_parts(deriver.inner_handle.clone(), deriver.inner_state.len()),
            ArrayArg::from_raw_parts(deriver.outer_handle.clone(), deriver.outer_state.len()),
            ArrayArg::from_raw_parts(
                deriver.bip32_inner_handle.clone(),
                deriver.bip32_inner_state.len(),
            ),
            ArrayArg::from_raw_parts(
                deriver.bip32_outer_handle.clone(),
                deriver.bip32_outer_state.len(),
            ),
            ArrayArg::from_raw_parts(expected_handle, expected.len()),
            ArrayArg::from_raw_parts(output_handle.clone(), candidates.len()),
            candidates.stride(),
        );
    }

    let output_bytes = client
        .read_one(output_handle)
        .map_err(|error| RecoverError::SeedDerivation(format!("CubeCL readback: {error:?}")))?;
    let matches = u32::from_bytes(&output_bytes)
        .iter()
        .enumerate()
        .filter_map(|(index, matched)| (*matched != 0).then_some(index))
        .collect();
    Ok(matches)
}

#[cube(launch)]
fn bip39_seed_kernel(
    candidates: &Array<u8>,
    lengths: &Array<u32>,
    inner_initial: &Array<u64>,
    outer_initial: &Array<u64>,
    output: &mut Array<u64>,
    stride: usize,
) {
    let candidate = ABSOLUTE_POS;
    if candidate < lengths.len() {
        let constants = Array::from_data(SHA512_CONSTANTS);
        let mut result = Array::<u64>::new(8usize);
        derive_bip39_seed(
            candidate,
            candidates,
            lengths,
            inner_initial,
            outer_initial,
            &constants,
            stride,
            &mut result,
        );
        let output_base = candidate * 8;
        let mut index = 0usize;
        while index < 8 {
            output[output_base + index] = result[index];
            index += 1;
        }
    }
}

#[cube(launch)]
fn bip39_chain_code_kernel(
    candidates: &Array<u8>,
    lengths: &Array<u32>,
    inner_initial: &Array<u64>,
    outer_initial: &Array<u64>,
    bip32_inner_initial: &Array<u64>,
    bip32_outer_initial: &Array<u64>,
    expected_chain_code: &Array<u64>,
    matches: &mut Array<u32>,
    stride: usize,
) {
    let candidate = ABSOLUTE_POS;
    if candidate < lengths.len() {
        let constants = Array::from_data(SHA512_CONSTANTS);
        let mut seed = Array::<u64>::new(8usize);
        derive_bip39_seed(
            candidate,
            candidates,
            lengths,
            inner_initial,
            outer_initial,
            &constants,
            stride,
            &mut seed,
        );
        let mut master = Array::<u64>::new(8usize);
        hmac_sha512_64(
            bip32_inner_initial,
            bip32_outer_initial,
            &seed,
            &constants,
            &mut master,
        );
        let mut matched = 1u32;
        let mut index = 0usize;
        while index < 4usize {
            if master[index + 4usize] != expected_chain_code[index] {
                matched = 0;
            }
            index += 1;
        }
        matches[candidate] = matched;
    }
}

#[cube]
fn derive_bip39_seed(
    candidate: usize,
    candidates: &Array<u8>,
    lengths: &Array<u32>,
    inner_initial: &Array<u64>,
    outer_initial: &Array<u64>,
    constants: &Array<u64>,
    stride: usize,
    result: &mut Array<u64>,
) {
    let length = usize::cast_from(lengths[candidate]);
    let mut first_data = Array::<u8>::new(112usize);
    first_data[0] = 0x6d;
    first_data[1] = 0x6e;
    first_data[2] = 0x65;
    first_data[3] = 0x6d;
    first_data[4] = 0x6f;
    first_data[5] = 0x6e;
    first_data[6] = 0x69;
    first_data[7] = 0x63;
    let base = candidate * stride;
    let mut offset = 0usize;
    while offset < length {
        first_data[8 + offset] = candidates[base + offset];
        offset += 1;
    }
    first_data[8 + length] = 0;
    first_data[9 + length] = 0;
    first_data[10 + length] = 0;
    first_data[11 + length] = 1;

    let mut current = Array::<u64>::new(8usize);
    hmac_sha512(
        inner_initial,
        outer_initial,
        constants,
        &first_data,
        length + 12,
        &mut current,
    );
    let mut word = 0usize;
    while word < 8usize {
        result[word] = current[word];
        word += 1;
    }

    let mut next = Array::<u64>::new(8usize);
    for _iteration in 1usize..2048usize {
        hmac_sha512_64(inner_initial, outer_initial, &current, constants, &mut next);
        let mut index = 0usize;
        while index < 8usize {
            current[index] = next[index];
            result[index] ^= next[index];
            index += 1;
        }
    }
}

#[cube]
fn hmac_sha512_64(
    inner_initial: &Array<u64>,
    outer_initial: &Array<u64>,
    data: &Array<u64>,
    constants: &Array<u64>,
    output: &mut Array<u64>,
) {
    let mut inner_hash = Array::<u64>::new(8usize);
    sha512_continue_64(inner_initial, data, constants, &mut inner_hash);
    sha512_continue_64(outer_initial, &inner_hash, constants, output);
}

#[cube]
fn sha512_continue_64(
    initial: &Array<u64>,
    data: &Array<u64>,
    constants: &Array<u64>,
    output: &mut Array<u64>,
) {
    let mut state = Array::<u64>::new(8usize);
    let mut index = 0usize;
    while index < 8usize {
        state[index] = initial[index];
        index += 1;
    }
    let mut schedule = Array::<u64>::new(16usize);
    index = 0;
    while index < 8usize {
        schedule[index] = data[index];
        index += 1;
    }
    schedule[8] = 0x8000000000000000u64;
    index = 9;
    while index < 15usize {
        schedule[index] = 0u64;
        index += 1;
    }
    schedule[15] = 1536u64;
    sha512_compress_schedule(&mut state, &mut schedule, constants);
    index = 0;
    while index < 8usize {
        output[index] = state[index];
        index += 1;
    }
}

#[cube]
fn hmac_sha512(
    inner_initial: &Array<u64>,
    outer_initial: &Array<u64>,
    constants: &Array<u64>,
    data: &Array<u8>,
    data_length: usize,
    output: &mut Array<u64>,
) {
    let mut inner_state = Array::<u64>::new(8usize);
    let mut index = 0usize;
    while index < 8 {
        inner_state[index] = inner_initial[index];
        index += 1;
    }
    sha512_continue(&mut inner_state, data, data_length, 128usize, constants);

    let mut inner_bytes = Array::<u8>::new(64usize);
    words_to_bytes(&inner_state, &mut inner_bytes);
    let mut outer_state = Array::<u64>::new(8usize);
    index = 0;
    while index < 8 {
        outer_state[index] = outer_initial[index];
        index += 1;
    }
    sha512_continue(&mut outer_state, &inner_bytes, 64usize, 128usize, constants);
    index = 0;
    while index < 8 {
        output[index] = outer_state[index];
        index += 1;
    }
}

#[cube]
fn sha512_continue(
    state: &mut Array<u64>,
    data: &Array<u8>,
    data_length: usize,
    prefix_length: usize,
    constants: &Array<u64>,
) {
    let mut blocks = 1usize;
    if data_length > 111usize {
        blocks = 2usize;
    }
    let bit_length = u64::cast_from(prefix_length + data_length) * 8u64;
    let mut block_index = 0usize;
    while block_index < blocks {
        let mut block = Array::<u8>::new(128usize);
        let mut offset = 0usize;
        while offset < 128 {
            let absolute = block_index * 128usize + offset;
            let mut value = 0u8;
            if absolute < data_length {
                value = data[absolute];
            } else if absolute == data_length {
                value = 0x80u8;
            } else if absolute >= blocks * 128usize - 8usize {
                let byte_index = absolute - (blocks * 128usize - 8usize);
                let shift = u64::cast_from((7usize - byte_index) * 8usize);
                value = u8::cast_from((bit_length >> shift) & 0xffu64);
            }
            block[offset] = value;
            offset += 1;
        }
        sha512_compress(state, &block, constants);
        block_index += 1;
    }
}

#[cube]
fn sha512_compress(state: &mut Array<u64>, block: &Array<u8>, constants: &Array<u64>) {
    let mut schedule = Array::<u64>::new(16usize);
    let mut index = 0usize;
    while index < 16 {
        let base = index * 8usize;
        schedule[index] = (u64::cast_from(block[base]) << 56u64)
            | (u64::cast_from(block[base + 1]) << 48u64)
            | (u64::cast_from(block[base + 2]) << 40u64)
            | (u64::cast_from(block[base + 3]) << 32u64)
            | (u64::cast_from(block[base + 4]) << 24u64)
            | (u64::cast_from(block[base + 5]) << 16u64)
            | (u64::cast_from(block[base + 6]) << 8u64)
            | u64::cast_from(block[base + 7]);
        index += 1;
    }

    sha512_compress_schedule(state, &mut schedule, constants);
}

#[cube]
fn sha512_compress_schedule(
    state: &mut Array<u64>,
    schedule: &mut Array<u64>,
    constants: &Array<u64>,
) {
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    let mut index = 0usize;
    while index < 80 {
        if index >= 16usize {
            let x = schedule[(index + 1usize) & 15usize];
            let y = schedule[(index + 14usize) & 15usize];
            let sigma0 = rotate_right(x, 1u64) ^ rotate_right(x, 8u64) ^ (x >> 7u64);
            let sigma1 = rotate_right(y, 19u64) ^ rotate_right(y, 61u64) ^ (y >> 6u64);
            schedule[index & 15usize] =
                schedule[index & 15usize] + sigma0 + schedule[(index + 9usize) & 15usize] + sigma1;
        }
        let sum1 = rotate_right(e, 14u64) ^ rotate_right(e, 18u64) ^ rotate_right(e, 41u64);
        let choice = (e & f) ^ ((!e) & g);
        let temporary1 = h + sum1 + choice + constants[index] + schedule[index & 15usize];
        let sum0 = rotate_right(a, 28u64) ^ rotate_right(a, 34u64) ^ rotate_right(a, 39u64);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
        index += 1;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

#[cube]
fn words_to_bytes(words: &Array<u64>, bytes: &mut Array<u8>) {
    let mut word = 0usize;
    while word < 8 {
        let value = words[word];
        let mut byte = 0usize;
        while byte < 8 {
            let shift = u64::cast_from((7usize - byte) * 8usize);
            bytes[word * 8usize + byte] = u8::cast_from((value >> shift) & 0xffu64);
            byte += 1;
        }
        word += 1;
    }
}

#[cube]
fn rotate_right(value: u64, amount: u64) -> u64 {
    let low = value >> amount;
    let high = value << (64u64 - amount);
    low | high
}

fn prepared_hmac_states(key: &str) -> ([u64; 8], [u64; 8]) {
    let mut key_block = [0_u8; 128];
    if key.len() > key_block.len() {
        let digest = Sha512::digest(key.as_bytes());
        key_block[..64].copy_from_slice(&digest);
    } else {
        key_block[..key.len()].copy_from_slice(key.as_bytes());
    }

    let mut inner_block = [0_u8; 128];
    let mut outer_block = [0_u8; 128];
    for index in 0..128 {
        inner_block[index] = key_block[index] ^ 0x36;
        outer_block[index] = key_block[index] ^ 0x5c;
    }
    let mut inner_state = SHA512_INITIAL;
    let mut outer_state = SHA512_INITIAL;
    sha512_compress_host(&mut inner_state, &inner_block);
    sha512_compress_host(&mut outer_state, &outer_block);
    (inner_state, outer_state)
}

fn sha512_compress_host(state: &mut [u64; 8], block: &[u8; 128]) {
    let mut schedule = [0_u64; 80];
    for (index, chunk) in block.chunks_exact(8).enumerate() {
        schedule[index] = u64::from_be_bytes(chunk.try_into().expect("eight-byte chunk"));
    }
    for index in 16..80 {
        let x = schedule[index - 15];
        let y = schedule[index - 2];
        let sigma0 = x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7);
        let sigma1 = y.rotate_right(19) ^ y.rotate_right(61) ^ (y >> 6);
        schedule[index] = schedule[index - 16]
            .wrapping_add(sigma0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(sigma1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..80 {
        let sum1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let choice = (e & f) ^ ((!e) & g);
        let temporary1 = h
            .wrapping_add(sum1)
            .wrapping_add(choice)
            .wrapping_add(SHA512_CONSTANTS[index])
            .wrapping_add(schedule[index]);
        let sum0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temporary2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temporary1);
        d = c;
        c = b;
        b = a;
        a = temporary1.wrapping_add(temporary2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

#[cfg(test)]
mod tests {
    use bip32::{Prefix, XPrv};
    use bip39::{Language, Mnemonic};

    use super::*;
    use crate::domain::{Candidate, MasterXpubTarget, TargetFingerprint};

    const PUBLIC_TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

    #[cfg(feature = "cube-cpu")]
    #[test]
    fn cube_cpu_matches_bip39_reference() {
        let secret = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let mut deriver = CubeSeedDeriver::cpu(&secret).unwrap();
        assert_matches_reference(&mut deriver);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_matches_bip39_reference() {
        let secret = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let mut deriver = CubeSeedDeriver::metal(&secret).unwrap();
        assert_matches_reference(&mut deriver);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_filters_master_chain_code_and_confirms_public_key() {
        let secret = SecretMnemonic::new(PUBLIC_TEST_MNEMONIC.to_owned());
        let mnemonic = Mnemonic::parse_in(Language::English, PUBLIC_TEST_MNEMONIC).unwrap();
        let expected = XPrv::new(mnemonic.to_seed("BenefitWIFE")).unwrap();
        let master_xpub =
            MasterXpubTarget::parse(&expected.public_key().to_string(Prefix::XPUB)).unwrap();
        let fingerprint = hex::encode(master_xpub.fingerprint())
            .parse::<TargetFingerprint>()
            .unwrap();
        let target = VerificationTarget::new(fingerprint, Some(master_xpub)).unwrap();
        let candidates = CandidateBatch::new(vec![
            Candidate::new(
                crate::domain::CandidateId("wrong".into()),
                crate::domain::SearchPhase::WrittenLower,
                vec!["wrong".into()],
            ),
            Candidate::new(
                crate::domain::CandidateId("match".into()),
                crate::domain::SearchPhase::WrittenCase,
                vec!["Benefit".into(), "WIFE".into()],
            ),
        ])
        .unwrap();
        let mut deriver = CubeSeedDeriver::metal(&secret).unwrap();

        assert_eq!(deriver.verify(&candidates, &target).unwrap(), [1]);
    }

    #[cfg(any(feature = "cube-cpu", all(feature = "metal", target_os = "macos")))]
    fn assert_matches_reference(deriver: &mut CubeSeedDeriver) {
        let candidates = vec![
            Candidate::new(
                crate::domain::CandidateId("empty".into()),
                crate::domain::SearchPhase::WrittenLower,
                vec![String::new()],
            ),
            Candidate::new(
                crate::domain::CandidateId("case".into()),
                crate::domain::SearchPhase::WrittenCase,
                vec!["Benefit".into(), "WIFE".into()],
            ),
        ];
        let batch = CandidateBatch::new(candidates).unwrap();
        let actual = deriver.derive_seeds(&batch).unwrap();
        let mnemonic = Mnemonic::parse_in(Language::English, PUBLIC_TEST_MNEMONIC).unwrap();

        assert_eq!(actual.as_slice()[0], mnemonic.to_seed(""));
        assert_eq!(actual.as_slice()[1], mnemonic.to_seed("BenefitWIFE"));
    }
}
