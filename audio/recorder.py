"""Record audio from the microphone."""

import math
import queue
import time
from typing import Callable

import numpy as np

import config
from audio.devices import require_sounddevice
from audio.exceptions import DeviceError
from audio.levels import level_bar, rms_dbfs
from audio.vad import EnergyVAD, SpeechSegmenter


def record(
    seconds: float = config.RECORD_SECONDS,
    sample_rate: int = config.SAMPLE_RATE,
    channels: int = config.CHANNELS,
    device=config.INPUT_DEVICE,
) -> np.ndarray:
    """Record a fixed number of seconds. Returns int16 audio of shape (frames, channels)."""
    if seconds <= 0:
        raise ValueError("seconds must be greater than 0")

    sd = require_sounddevice()
    frames = int(seconds * sample_rate)
    try:
        # Check first, so we get a clear error if the mic does not support these settings.
        sd.check_input_settings(device=device, samplerate=sample_rate,
                                channels=channels, dtype=config.DTYPE)
        audio = sd.rec(frames, samplerate=sample_rate, channels=channels,
                       dtype=config.DTYPE, device=device)
        sd.wait()  # block until recording is finished
    except (sd.PortAudioError, ValueError) as exc:
        raise DeviceError(f"Could not record from device {device!r}: {exc}") from exc
    return audio


def monitor_levels(
    seconds: float = 10,
    sample_rate: int = config.SAMPLE_RATE,
    device=config.INPUT_DEVICE,
) -> float:
    """Show a live loudness bar for the mic. Returns the loudest level seen (dBFS).

    Uses a stream with a callback: sounddevice calls `callback` many times per second
    with a small block of new audio. We only store the level there (callbacks must be fast)
    and print it from the main loop.
    """
    sd = require_sounddevice()
    state = {"db": -math.inf, "max_db": -math.inf, "problems": 0}

    def callback(indata, frames, time_info, status):
        if status:  # e.g. input overflow = we were too slow to read audio
            state["problems"] += 1
        db = rms_dbfs(indata)
        state["db"] = db
        state["max_db"] = max(state["max_db"], db)

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype=config.DTYPE,
                            device=device, callback=callback):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                print("\r" + level_bar(state["db"]), end="", flush=True)
                time.sleep(0.05)
    except (sd.PortAudioError, ValueError) as exc:
        raise DeviceError(f"Could not open mic {device!r}: {exc}") from exc
    finally:
        print()

    if state["problems"]:
        print(f"Note: {state['problems']} stream warnings (audio blocks were dropped).")
    return state["max_db"]


def record_until_silence(
    max_seconds: float = config.MAX_RECORD_SECONDS,
    silence_seconds: float = config.SILENCE_SECONDS,
    start_timeout: float = config.START_TIMEOUT_SECONDS,
    sample_rate: int = config.SAMPLE_RATE,
    device=config.INPUT_DEVICE,
    on_event: Callable[[str], None] | None = None,
) -> tuple[np.ndarray | None, str]:
    """Record until the speaker stops talking.

    Returns (audio, stop_reason). audio is 1-D int16, or None if nobody spoke.
    stop_reason is "silence", "no_speech" or "max_length".
    on_event(name) is called with "calibrated" and "speech_start" so the UI can react.

    How it works: the stream callback only copies each small chunk into a queue
    (callbacks must be quick). The main loop takes chunks out of the queue and
    gives them to the SpeechSegmenter, which decides when to stop.
    """
    sd = require_sounddevice()
    frame_length = int(sample_rate * config.VAD_FRAME_MS / 1000)
    vad = EnergyVAD(
        calibration_frames=max(int(config.VAD_CALIBRATION_SECONDS * 1000 / config.VAD_FRAME_MS), 1),
        margin_db=config.VAD_MARGIN_DB,
        min_threshold_db=config.VAD_MIN_THRESHOLD_DBFS,
        max_threshold_db=config.VAD_MAX_THRESHOLD_DBFS,
    )
    segmenter = SpeechSegmenter(
        vad, sample_rate, frame_length,
        silence_seconds=silence_seconds, start_timeout=start_timeout,
        max_seconds=max_seconds, pre_roll_seconds=config.PRE_ROLL_SECONDS,
    )
    chunks: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        chunks.put(indata[:, 0].copy())

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype=config.DTYPE,
                            device=device, blocksize=frame_length, callback=callback):
            while not segmenter.done:
                try:
                    chunk = chunks.get(timeout=2.0)
                except queue.Empty:
                    raise DeviceError("The microphone stopped sending audio.") from None
                event = segmenter.feed(chunk)
                if event and on_event and event != "done":
                    on_event(event)
    except (sd.PortAudioError, ValueError) as exc:
        raise DeviceError(f"Could not record from device {device!r}: {exc}") from exc

    return segmenter.result(), segmenter.stop_reason
