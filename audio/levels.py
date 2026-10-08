"""Measure how loud audio is. Useful to check if the mic works and is set up well."""

import math

import numpy as np

INT16_MAX = 32767


def rms_dbfs(audio: np.ndarray) -> float:
    """Average loudness in dBFS. 0 = max possible, -inf = total silence.

    Works with int16 audio or float audio in the range -1.0 to 1.0.

    Normal speech near a mic is usually between -30 and -10 dBFS.
    """
    if audio.size == 0:
        return -math.inf
    if np.issubdtype(audio.dtype, np.floating):
        samples = audio.astype(np.float32)            # already in -1.0 .. 1.0
    else:
        samples = audio.astype(np.float32) / 32768.0  # int16 -> -1.0 .. 1.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return 20 * math.log10(rms) if rms > 0 else -math.inf


def audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    """Return duration, peak level, average level and number of clipped samples."""
    if audio.size == 0:
        return {"duration_s": 0.0, "peak_dbfs": -math.inf, "rms_dbfs": -math.inf, "clipped": 0}

    abs_audio = np.abs(audio.astype(np.int32))
    peak = int(abs_audio.max())
    return {
        "duration_s": audio.shape[0] / sample_rate,
        "peak_dbfs": 20 * math.log10(peak / 32768.0) if peak > 0 else -math.inf,
        "rms_dbfs": rms_dbfs(audio),
        # Clipping = sound too loud, samples hit the limit and get distorted.
        "clipped": int(np.sum(abs_audio >= INT16_MAX)),
    }


def level_bar(db: float, width: int = 40, floor_db: float = -60.0) -> str:
    """Turn a dB value into a text bar like: [#########.........]  -23.4 dBFS"""
    if math.isinf(db):
        fraction = 0.0
    else:
        fraction = min(max((db - floor_db) / -floor_db, 0.0), 1.0)
    filled = int(fraction * width)
    label = "  -inf" if math.isinf(db) else f"{db:6.1f}"
    return f"[{'#' * filled}{'.' * (width - filled)}] {label} dBFS"
