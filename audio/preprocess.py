"""Get audio ready for Whisper.

Whisper wants: float32 numbers between -1.0 and 1.0, one channel (mono), 16 kHz.
Our recordings are int16 and may come from files with other rates or stereo,
so these small functions convert everything into that one format.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from audio.levels import rms_dbfs

WHISPER_SAMPLE_RATE = 16000


def to_float32(audio: np.ndarray) -> np.ndarray:
    """int16 (-32768..32767) -> float32 (-1.0..1.0). Float input is kept as is."""
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)
    if audio.dtype != np.int16:
        raise ValueError(f"Unsupported audio type: {audio.dtype}")
    return audio.astype(np.float32) / 32768.0


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return 1-D audio. Stereo (frames, 2) is mixed down by averaging the channels."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    raise ValueError(f"Expected 1-D or 2-D audio, got shape {audio.shape}")


def resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Change the sample rate, e.g. 48000 Hz -> 16000 Hz.

    resample_poly filters out sounds too high for the new rate before
    dropping samples. Simply taking every 3rd sample would create noise (aliasing).
    """
    if orig_rate == target_rate:
        return audio
    divisor = gcd(orig_rate, target_rate)
    up, down = target_rate // divisor, orig_rate // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int,
                 threshold_dbfs: float = -45.0, pad_seconds: float = 0.2) -> np.ndarray:
    """Cut quiet parts from the start and end. Keeps a little padding around speech.

    Returns an empty array if the whole clip is quieter than the threshold.
    """
    frame = max(int(sample_rate * 0.03), 1)  # 30 ms chunks
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    loud = [i for i in range(n_frames)
            if rms_dbfs(audio[i * frame:(i + 1) * frame]) > threshold_dbfs]
    if not loud:
        return audio[:0]
    pad = int(pad_seconds * sample_rate)
    start = max(loud[0] * frame - pad, 0)
    end = min((loud[-1] + 1) * frame + pad, len(audio))
    return audio[start:end]


def normalize_peak(audio: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Make the loudest point hit target_dbfs. Silent audio is returned unchanged.

    Whisper handles different volumes quite well on its own, so this is optional.
    It helps most with very quiet recordings.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-4:  # basically silence: boosting it would only boost noise
        return audio
    gain = (10 ** (target_dbfs / 20)) / peak
    return (audio * gain).astype(np.float32)


def prepare_for_whisper(audio: np.ndarray, sample_rate: int,
                        trim: bool = False, normalize: bool = False) -> np.ndarray:
    """Full pipeline: any audio -> float32, mono, 16 kHz (+ optional trim/normalize)."""
    samples = to_mono(to_float32(audio))
    samples = resample(samples, sample_rate, WHISPER_SAMPLE_RATE)
    if trim:
        samples = trim_silence(samples, WHISPER_SAMPLE_RATE)
    if normalize:
        samples = normalize_peak(samples)
    return np.ascontiguousarray(samples, dtype=np.float32)
