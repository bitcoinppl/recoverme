"""JAX device discovery and explicit backend selection."""

from __future__ import annotations

import jax

from recoverme_jax.domain import Backend, DeviceInfo


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested backend is unavailable."""


def available_devices() -> tuple[tuple[DeviceInfo, jax.Device], ...]:
    """Return the supported local CPU and NVIDIA CUDA devices."""
    discovered = [(_describe(Backend.CPU, device), device) for device in _devices_for("cpu")]
    for device in _devices_for("gpu"):
        kind = device.device_kind.lower()
        if "nvidia" in kind or "cuda" in kind:
            discovered.append((_describe(Backend.CUDA, device), device))
    return tuple(discovered)


def select_device(backend: Backend, device_index: int = 0) -> tuple[DeviceInfo, jax.Device]:
    """Select one supported JAX device without silently changing explicit backends."""
    devices = available_devices()
    requested = (
        Backend.CUDA
        if backend is Backend.AUTO and any(info.backend is Backend.CUDA for info, _ in devices)
        else Backend.CPU
        if backend is Backend.AUTO
        else backend
    )
    matching = [(info, device) for info, device in devices if info.backend is requested]
    if device_index < 0 or device_index >= len(matching):
        if requested is Backend.CUDA:
            raise BackendUnavailableError(
                "CUDA backend requested but JAX detected no matching NVIDIA device"
            )
        raise BackendUnavailableError(
            f"{requested.value} device index {device_index} is unavailable"
        )
    return matching[device_index]


def _devices_for(backend: str) -> tuple[jax.Device, ...]:
    try:
        return tuple(jax.devices(backend))
    except RuntimeError:
        return ()


def _describe(backend: Backend, device: jax.Device) -> DeviceInfo:
    return DeviceInfo(
        backend=backend,
        platform=device.platform,
        device_kind=device.device_kind,
        device_id=device.id,
        jax_version=jax.__version__,
    )
