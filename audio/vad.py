"""Voice Activity Detection (VAD): decide when someone is talking.

This is a simple energy-based VAD: a chunk of audio counts as "speech"
if it is clearly louder than the background noise of the room.

It's not perfect (a loud fan or music can fool it), but it's easy to
understand and good enough to stop recording when you stop talking.
Later we could swap it for a neural VAD like Silero.
"""

from collections import deque

import numpy as np

from audio.levels import rms_dbfs

# Segmenter states
WAITING = "waiting"    # listening, no speech yet
SPEAKING = "speaking"  # speech started, collecting audio
DONE = "done"          # finished (see stop_reason)


class EnergyVAD:
    """Says True for chunks louder than (room noise + margin).

    The first `calibration_frames` chunks are used to measure the room noise,
    so the user should stay quiet for a moment at the start.
    """

    def __init__(self, calibration_frames: int = 15, margin_db: float = 10.0,
                 min_threshold_db: float = -50.0, max_threshold_db: float = -25.0):
        self.calibration_frames = calibration_frames
        self.margin_db = margin_db
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.threshold_db = None
        self.noise_db = None
        self._noise_levels = []

    @property
    def calibrated(self) -> bool:
        return self.threshold_db is not None

    def is_speech(self, frame: np.ndarray) -> bool:
        db = rms_dbfs(frame)
        if not self.calibrated:
            self._noise_levels.append(max(db, -100.0))  # -inf (digital silence) -> -100
            if len(self._noise_levels) >= self.calibration_frames:
                # Median ignores a single click or bump during calibration.
                self.noise_db = float(np.median(self._noise_levels))
                raw = self.noise_db + self.margin_db
                self.threshold_db = min(max(raw, self.min_threshold_db), self.max_threshold_db)
            return False
        return db > self.threshold_db


class SpeechSegmenter:
    """Feed it audio chunks one by one; it decides when an utterance starts and ends.

    This class does not touch the microphone. That keeps it easy to test:
    we can feed it fake audio and check what it decides.
    """

    def __init__(self, vad: EnergyVAD, sample_rate: int, frame_length: int,
                 silence_seconds: float = 1.2, start_timeout: float = 8.0,
                 max_seconds: float = 60.0, pre_roll_seconds: float = 0.3,
                 start_frames: int = 3):
        self.vad = vad
        self.frame_seconds = frame_length / sample_rate
        self.silence_seconds = silence_seconds
        self.start_timeout = start_timeout
        self.max_seconds = max_seconds
        # Speech must last a few chunks in a row to count (ignores clicks and taps).
        self.start_frames = start_frames
        pre_roll_len = max(round(pre_roll_seconds / self.frame_seconds), start_frames)
        self._pre_roll = deque(maxlen=pre_roll_len)

        self._frames = []
        self._speech_run = 0
        self._silence_run = 0.0
        self._elapsed = 0.0
        self.state = WAITING
        self.stop_reason = None  # "silence", "no_speech" or "max_length"

    @property
    def done(self) -> bool:
        return self.state == DONE

    def _finish(self, reason: str) -> None:
        self.state = DONE
        self.stop_reason = reason

    def feed(self, frame: np.ndarray) -> str | None:
        """Process one chunk. Returns an event name when something changes, else None.

        Events: "calibrated", "speech_start", "done".
        """
        if self.done:
            return None

        was_calibrated = self.vad.calibrated
        self._elapsed += self.frame_seconds
        speech = self.vad.is_speech(frame)
        event = "calibrated" if self.vad.calibrated and not was_calibrated else None

        if self.state == WAITING:
            # Keep recent chunks so the start of the first word isn't cut off.
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self.start_frames:
                self.state = SPEAKING
                self._frames = list(self._pre_roll)
                self._pre_roll.clear()
                event = "speech_start"
            elif self._elapsed >= self.start_timeout:
                self._finish("no_speech")
                return "done"

        elif self.state == SPEAKING:
            self._frames.append(frame)
            self._silence_run = 0.0 if speech else self._silence_run + self.frame_seconds
            if self._silence_run >= self.silence_seconds:
                self._finish("silence")
                return "done"

        if self.state == SPEAKING and self._elapsed >= self.max_seconds:
            self._finish("max_length")
            return "done"
        return event

    def result(self) -> np.ndarray | None:
        """The recorded speech as one array, or None if no speech was heard."""
        if not self._frames:
            return None
        frames = self._frames
        # Drop most of the trailing silence, but keep ~0.2 s so the end sounds natural.
        extra = int(max(self._silence_run - 0.2, 0.0) / self.frame_seconds)
        if extra and extra < len(frames):
            frames = frames[:-extra]
        return np.concatenate(frames)
