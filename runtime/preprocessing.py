from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np


def decode_emg_packet(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode one Myo EMG notification into two signed 8-channel samples."""
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    if raw.size != 16:
        raise ValueError(f"expected 16 EMG bytes, got {raw.size}")

    signed = raw.astype(np.int16)
    signed[signed > 127] -= 256
    return signed.reshape(2, 8)


def normalize_window(
    window: np.ndarray,
    scaling_params: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Apply per-channel z-score normalization to an 8 x N EMG window."""
    array = np.asarray(window, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D window, got shape {array.shape}")

    normalized = array.copy()
    for channel in range(array.shape[0]):
        key = str(channel)
        if key not in scaling_params:
            raise KeyError(f"missing scaling parameters for channel {channel}")

        mean, std = scaling_params[key]
        if float(std) == 0:
            raise ValueError(f"channel {channel} has zero standard deviation")
        normalized[channel] = (normalized[channel] - float(mean)) / float(std)

    return normalized


def prepare_model_input(
    window: np.ndarray,
    scaling_params: Mapping[str, Sequence[float]],
    *,
    channels: int = 8,
    samples: int = 24,
) -> np.ndarray:
    """Return a float32 ONNX input with shape [1, 1, channels, samples]."""
    array = np.asarray(window)
    expected = (channels, samples)
    if array.shape != expected:
        raise ValueError(f"expected window shape {expected}, got {array.shape}")

    normalized = normalize_window(array, scaling_params)
    return normalized[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)


def majority_vote(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot vote over an empty sequence")
    return Counter(values).most_common(1)[0][0]
