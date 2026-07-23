"""Batched JIT-compiled BIP39 PBKDF2-HMAC-SHA512 implemented with JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

MAX_MNEMONIC_BYTES: Final = 239
MAX_PASSPHRASE_BYTES: Final = 100
PBKDF2_ROUNDS: Final = 2_048

type U8Array = jax.Array
type U64Array = jax.Array

_INITIAL: Final = jnp.asarray(
    [
        0x6A09E667F3BCC908,
        0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B,
        0xA54FF53A5F1D36F1,
        0x510E527FADE682D1,
        0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B,
        0x5BE0CD19137E2179,
    ],
    dtype=jnp.uint64,
)

_CONSTANTS: Final = jnp.asarray(
    [
        0x428A2F98D728AE22,
        0x7137449123EF65CD,
        0xB5C0FBCFEC4D3B2F,
        0xE9B5DBA58189DBBC,
        0x3956C25BF348B538,
        0x59F111F1B605D019,
        0x923F82A4AF194F9B,
        0xAB1C5ED5DA6D8118,
        0xD807AA98A3030242,
        0x12835B0145706FBE,
        0x243185BE4EE4B28C,
        0x550C7DC3D5FFB4E2,
        0x72BE5D74F27B896F,
        0x80DEB1FE3B1696B1,
        0x9BDC06A725C71235,
        0xC19BF174CF692694,
        0xE49B69C19EF14AD2,
        0xEFBE4786384F25E3,
        0x0FC19DC68B8CD5B5,
        0x240CA1CC77AC9C65,
        0x2DE92C6F592B0275,
        0x4A7484AA6EA6E483,
        0x5CB0A9DCBD41FBD4,
        0x76F988DA831153B5,
        0x983E5152EE66DFAB,
        0xA831C66D2DB43210,
        0xB00327C898FB213F,
        0xBF597FC7BEEF0EE4,
        0xC6E00BF33DA88FC2,
        0xD5A79147930AA725,
        0x06CA6351E003826F,
        0x142929670A0E6E70,
        0x27B70A8546D22FFC,
        0x2E1B21385C26C926,
        0x4D2C6DFC5AC42AED,
        0x53380D139D95B3DF,
        0x650A73548BAF63DE,
        0x766A0ABB3C77B2A8,
        0x81C2C92E47EDAEE6,
        0x92722C851482353B,
        0xA2BFE8A14CF10364,
        0xA81A664BBC423001,
        0xC24B8B70D0F89791,
        0xC76C51A30654BE30,
        0xD192E819D6EF5218,
        0xD69906245565A910,
        0xF40E35855771202A,
        0x106AA07032BBD1B8,
        0x19A4C116B8D2D0C8,
        0x1E376C085141AB53,
        0x2748774CDF8EEB99,
        0x34B0BCB5E19B48A8,
        0x391C0CB3C5C95A63,
        0x4ED8AA4AE3418ACB,
        0x5B9CCA4F7763E373,
        0x682E6FF3D6B2B8A3,
        0x748F82EE5DEFB2FC,
        0x78A5636F43172F60,
        0x84C87814A1F0AB72,
        0x8CC702081A6439EC,
        0x90BEFFFA23631E28,
        0xA4506CEBDE82BDE9,
        0xBEF9A3F7B2C67915,
        0xC67178F2E372532B,
        0xCA273ECEEA26619C,
        0xD186B8C721C0C207,
        0xEADA7DD6CDE0EB1E,
        0xF57D4F7FEE6ED178,
        0x06F067AA72176FBA,
        0x0A637DC5A2C898A6,
        0x113F9804BEF90DAE,
        0x1B710B35131C471B,
        0x28DB77F523047D84,
        0x32CAAB7B40C72493,
        0x3C9EBE0A15C9BEBC,
        0x431D67C49C100D4C,
        0x4CC5D4BECB3E42B6,
        0x597F299CFC657E2A,
        0x5FCB6FAB3AD6FAEC,
        0x6C44198C4A475817,
    ],
    dtype=jnp.uint64,
)


def _rotate_right(value: U64Array, amount: int) -> U64Array:
    return (value >> np.uint64(amount)) | (value << np.uint64(64 - amount))


def _sha512_compress(state: U64Array, block: U8Array) -> U64Array:
    state = jnp.broadcast_to(state, (*block.shape[:-1], 8))
    chunks = block.reshape((*block.shape[:-1], 16, 8)).astype(jnp.uint64)
    shifts = jnp.asarray([56, 48, 40, 32, 24, 16, 8, 0], dtype=jnp.uint64)
    first = jnp.bitwise_or.reduce(chunks << shifts, axis=-1)
    schedule = jnp.zeros((*block.shape[:-1], 80), dtype=jnp.uint64)
    schedule = schedule.at[..., :16].set(first)

    def extend(index: int, words: U64Array) -> U64Array:
        x = words[..., index - 15]
        y = words[..., index - 2]
        sigma0 = _rotate_right(x, 1) ^ _rotate_right(x, 8) ^ (x >> np.uint64(7))
        sigma1 = _rotate_right(y, 19) ^ _rotate_right(y, 61) ^ (y >> np.uint64(6))
        value = words[..., index - 16] + sigma0 + words[..., index - 7] + sigma1
        return words.at[..., index].set(value)

    schedule = lax.fori_loop(16, 80, extend, schedule)
    working = tuple(state[..., index] for index in range(8))

    def compress_round(index: int, values: tuple[U64Array, ...]) -> tuple[U64Array, ...]:
        a, b, c, d, e, f, g, h = values
        sum1 = _rotate_right(e, 14) ^ _rotate_right(e, 18) ^ _rotate_right(e, 41)
        choice = (e & f) ^ ((~e) & g)
        temporary1 = h + sum1 + choice + _CONSTANTS[index] + schedule[..., index]
        sum0 = _rotate_right(a, 28) ^ _rotate_right(a, 34) ^ _rotate_right(a, 39)
        majority = (a & b) ^ (a & c) ^ (b & c)
        temporary2 = sum0 + majority
        return (
            temporary1 + temporary2,
            a,
            b,
            c,
            d + temporary1,
            e,
            f,
            g,
        )

    working = lax.fori_loop(0, 80, compress_round, working)
    return state + jnp.stack(working, axis=-1)


def _length_bytes(bit_lengths: U64Array) -> U8Array:
    shifts = jnp.asarray([56, 48, 40, 32, 24, 16, 8, 0], dtype=jnp.uint64)
    return ((bit_lengths[..., None] >> shifts) & np.uint64(0xFF)).astype(jnp.uint8)


def _mnemonic_digest(mnemonic: U8Array, length: U64Array) -> U8Array:
    positions = jnp.arange(128, dtype=jnp.uint64)
    safe = jnp.minimum(positions, np.uint64(MAX_MNEMONIC_BYTES - 1)).astype(jnp.int32)
    block0 = jnp.where(positions < length, mnemonic[safe], np.uint8(0))
    block0 = jnp.where(positions == length, np.uint8(0x80), block0)
    bit_length = length * np.uint64(8)
    one_block = length <= np.uint64(111)
    encoded = _length_bytes(bit_length)
    block0 = jnp.where(
        one_block & (positions >= np.uint64(120)),
        encoded[(positions - np.uint64(120)).clip(0, 7).astype(jnp.int32)],
        block0,
    )
    absolute = positions + np.uint64(128)
    safe_second = jnp.minimum(absolute, np.uint64(MAX_MNEMONIC_BYTES - 1)).astype(jnp.int32)
    block1 = jnp.where(absolute < length, mnemonic[safe_second], np.uint8(0))
    block1 = jnp.where(absolute == length, np.uint8(0x80), block1)
    block1 = jnp.where(
        positions >= np.uint64(120),
        encoded[(positions - np.uint64(120)).clip(0, 7).astype(jnp.int32)],
        block1,
    )
    first = _sha512_compress(_INITIAL, block0)
    second = _sha512_compress(first, block1)
    return _words_to_bytes(jnp.where(one_block, first, second))


def _prepared_hmac_states(mnemonic: U8Array, length: U64Array) -> tuple[U64Array, U64Array]:
    digest = _mnemonic_digest(mnemonic, length)
    positions = jnp.arange(128, dtype=jnp.uint64)
    safe = jnp.minimum(positions, np.uint64(MAX_MNEMONIC_BYTES - 1)).astype(jnp.int32)
    short_key = jnp.where(positions < length, mnemonic[safe], np.uint8(0))
    digest_index = positions.clip(0, 63).astype(jnp.int32)
    long_key = jnp.where(positions < np.uint64(64), digest[digest_index], np.uint8(0))
    key = jnp.where(length > np.uint64(128), long_key, short_key)
    inner = _sha512_compress(_INITIAL, key ^ np.uint8(0x36))
    outer = _sha512_compress(_INITIAL, key ^ np.uint8(0x5C))
    return inner, outer


def _continue_one_block(initial: U64Array, data: U8Array, lengths: U64Array) -> U64Array:
    positions = jnp.arange(128, dtype=jnp.uint64)
    safe = jnp.minimum(positions, np.uint64(data.shape[-1] - 1)).astype(jnp.int32)
    gathered = jnp.take(data, safe, axis=-1)
    block = jnp.where(positions < lengths[..., None], gathered, np.uint8(0))
    block = jnp.where(positions == lengths[..., None], np.uint8(0x80), block)
    encoded = _length_bytes((np.uint64(128) + lengths) * np.uint64(8))
    encoded_index = (positions - np.uint64(120)).clip(0, 7).astype(jnp.int32)
    encoded_values = jnp.take(encoded, encoded_index, axis=-1)
    block = jnp.where(
        (lengths <= np.uint64(111))[..., None] & (positions >= np.uint64(120)),
        encoded_values,
        block,
    )
    return _sha512_compress(initial, block)


def _continue_salt(initial: U64Array, data: U8Array, lengths: U64Array) -> U64Array:
    first = _continue_one_block(initial, data, lengths)
    positions = jnp.arange(128, dtype=jnp.uint64)
    second_block = jnp.zeros((*lengths.shape, 128), dtype=jnp.uint8)
    encoded = _length_bytes((np.uint64(128) + lengths) * np.uint64(8))
    encoded_index = (positions - np.uint64(120)).clip(0, 7).astype(jnp.int32)
    encoded_values = jnp.take(encoded, encoded_index, axis=-1)
    second_block = jnp.where(positions >= np.uint64(120), encoded_values, second_block)
    second = _sha512_compress(first, second_block)
    return jnp.where((lengths > np.uint64(111))[..., None], second, first)


def _words_to_bytes(words: U64Array) -> U8Array:
    shifts = jnp.asarray([56, 48, 40, 32, 24, 16, 8, 0], dtype=jnp.uint64)
    return (
        ((words[..., None] >> shifts) & np.uint64(0xFF))
        .astype(jnp.uint8)
        .reshape((*words.shape[:-1], 64))
    )


def _hmac_one(
    inner_initial: U64Array,
    outer_initial: U64Array,
    data: U8Array,
    lengths: U64Array,
) -> U64Array:
    inner = _continue_one_block(inner_initial, data, lengths)
    return _continue_one_block(
        outer_initial,
        _words_to_bytes(inner),
        jnp.full(lengths.shape, 64, dtype=jnp.uint64),
    )


def _pbkdf2_batch(
    mnemonic: U8Array,
    mnemonic_length: U64Array,
    candidates: U8Array,
    lengths: U64Array,
) -> U8Array:
    inner_initial, outer_initial = _prepared_hmac_states(mnemonic, mnemonic_length)
    positions = jnp.arange(112, dtype=jnp.uint64)
    prefix = jnp.asarray(np.frombuffer(b"mnemonic", dtype=np.uint8))
    prefix_values = prefix[jnp.minimum(positions, np.uint64(7)).astype(jnp.int32)]
    candidate_positions = (positions - np.uint64(8)).clip(0, MAX_PASSPHRASE_BYTES - 1)
    candidate_values = jnp.take(candidates, candidate_positions.astype(jnp.int32), axis=-1)
    salt = jnp.where(positions < np.uint64(8), prefix_values, np.uint8(0))
    salt = jnp.where(
        (positions >= np.uint64(8)) & (positions < lengths[..., None] + np.uint64(8)),
        candidate_values,
        salt,
    )
    salt = jnp.where(positions == lengths[..., None] + np.uint64(11), np.uint8(1), salt)
    salt_lengths = lengths + np.uint64(12)
    first_inner = _continue_salt(inner_initial, salt, salt_lengths)
    current = _continue_one_block(
        outer_initial,
        _words_to_bytes(first_inner),
        jnp.full(lengths.shape, 64, dtype=jnp.uint64),
    )

    def iterate(_: int, values: tuple[U64Array, U64Array]) -> tuple[U64Array, U64Array]:
        previous, result = values
        next_words = _hmac_one(
            inner_initial,
            outer_initial,
            _words_to_bytes(previous),
            jnp.full(lengths.shape, 64, dtype=jnp.uint64),
        )
        return next_words, result ^ next_words

    _, result = lax.fori_loop(1, PBKDF2_ROUNDS, iterate, (current, current))
    return _words_to_bytes(result)


@dataclass(slots=True)
class JaxSeedDeriver:
    """Compiled fixed-shape seed deriver bound to one JAX device."""

    device: jax.Device
    batch_size: int
    _mnemonic: jax.Array
    _mnemonic_length: jax.Array
    _compiled: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]

    @classmethod
    def create(
        cls,
        mnemonic: np.ndarray[tuple[int], np.dtype[np.uint8]],
        device: jax.Device,
        batch_size: int,
    ) -> JaxSeedDeriver:
        """Create a deriver and transfer mutable mnemonic material to its device."""
        if mnemonic.ndim != 1 or mnemonic.size > MAX_MNEMONIC_BYTES:
            raise ValueError("normalized mnemonic length is unsupported")
        if mnemonic.dtype != np.uint8:
            raise ValueError("mnemonic buffer must use uint8 elements")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        padded = np.zeros(MAX_MNEMONIC_BYTES, dtype=np.uint8)
        padded[: mnemonic.size] = mnemonic
        mnemonic_length = np.uint64(mnemonic.size)
        with jax.default_device(device):
            device_mnemonic = jax.device_put(padded, device).copy()
            device_mnemonic.block_until_ready()
            device_length = jax.device_put(mnemonic_length, device)
            compiled = cast(
                "Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]",
                jax.jit(_pbkdf2_batch),
            )
        padded.fill(0)
        mnemonic.fill(0)
        return cls(device, batch_size, device_mnemonic, device_length, compiled)

    def derive(
        self,
        candidate_bytes: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        lengths: np.ndarray[tuple[int], np.dtype[np.uint16]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        """Derive one 64-byte seed for every fixed-width batch row."""
        expected = (self.batch_size, MAX_PASSPHRASE_BYTES)
        if candidate_bytes.shape != expected or lengths.shape != (self.batch_size,):
            raise ValueError("candidate buffers do not match the compiled batch shape")
        if candidate_bytes.dtype != np.uint8 or lengths.dtype != np.uint16:
            raise ValueError("candidate buffers use unsupported element types")
        if np.any(lengths > MAX_PASSPHRASE_BYTES):
            raise ValueError("candidate length exceeds the supported maximum")
        with jax.default_device(self.device):
            result = self._compiled(
                self._mnemonic,
                self._mnemonic_length,
                jax.device_put(candidate_bytes, self.device),
                jax.device_put(lengths.astype(np.uint64), self.device),
            )
        result.block_until_ready()
        return np.asarray(result, dtype=np.uint8).copy()
