"""Speech-to-text with faster-whisper.

faster-whisper runs OpenAI's Whisper models with a faster engine (CTranslate2).
It gives the same text as the original Whisper, but is several times faster
and uses less memory, which matters a lot on a CPU.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from audio.preprocess import WHISPER_SAMPLE_RATE, prepare_for_whisper
from audio.wav_io import load_wav
from stt.exceptions import STTError


@dataclass
class Segment:
    """One piece of the transcript (Whisper splits speech into phrases)."""
    start: float           # seconds from start of audio
    end: float
    text: str
    avg_logprob: float     # how sure the model was (closer to 0 = more sure)
    no_speech_prob: float  # model's guess that this part is not speech at all


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    language_probability: float
    audio_seconds: float
    processing_seconds: float
    segments: list[Segment] = field(default_factory=list)
    dropped_segments: int = 0

    @property
    def rtf(self) -> float:
        """Real-Time Factor = processing time / audio length.

        0.2 means 10 s of audio took 2 s. Below 1.0 is faster than real time.
        """
        return self.processing_seconds / self.audio_seconds if self.audio_seconds else 0.0


class WhisperSTT:
    """Loads a Whisper model once and transcribes audio with it.

    Loading takes a few seconds, so create one WhisperSTT and reuse it.
    """

    def __init__(self, model_size: str = config.WHISPER_MODEL,
                 device: str = config.WHISPER_DEVICE,
                 compute_type: str = config.WHISPER_COMPUTE_TYPE,
                 language: str | None = config.WHISPER_LANGUAGE,
                 beam_size: int = config.WHISPER_BEAM_SIZE,
                 vad_filter: bool = config.WHISPER_VAD_FILTER,
                 initial_prompt: str | None = config.WHISPER_INITIAL_PROMPT,
                 model=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        self.load_seconds = 0.0
        self._model = model  # tests can pass a fake model here

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model (downloads it into models/whisper/ the first time)."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        # except ImportError as exc:
        #     raise STTError("faster-whisper is not installed. Run: pip install faster-whisper") from exc
        except ImportError as exc:
            raise STTError(
                f"Could not import faster-whisper. Check its dependencies. Actual error: {exc}"
            ) from exc

        start = time.perf_counter()
        try:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type,
                download_root=str(config.MODELS_DIR / "whisper"),
            )
        # faster-whisper can fail in many ways (no internet on first run, bad model name,
        # not enough memory, no GPU). We turn all of them into one clear error.
        except Exception as exc:
            raise STTError(f"Could not load Whisper model '{self.model_size}': {exc}") from exc
        self.load_seconds = time.perf_counter() - start

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptResult:
        """Turn audio (int16 or float, any rate, mono or stereo) into text."""
        self.load()
        samples = prepare_for_whisper(audio, sample_rate)
        audio_seconds = len(samples) / WHISPER_SAMPLE_RATE

        if audio_seconds < 0.1:  # too short to contain any words
            return TranscriptResult("", self.language, 0.0, audio_seconds, 0.0)

        start = time.perf_counter()
        try:
            segments_iter, info = self._model.transcribe(
                samples,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                initial_prompt=self.initial_prompt,
                # Don't feed earlier text back in. Helps stop repeated/looping text.
                condition_on_previous_text=False,
            )
            # segments_iter is a generator: the real work happens while we loop over it.
            raw = list(segments_iter)
        except Exception as exc:
            raise STTError(f"Transcription failed: {exc}") from exc
        processing_seconds = time.perf_counter() - start

        kept, dropped = [], 0
        for s in raw:
            seg = Segment(s.start, s.end, s.text.strip(), s.avg_logprob, s.no_speech_prob)
            if self._looks_like_silence(seg) or not seg.text:
                dropped += 1
            else:
                kept.append(seg)

        return TranscriptResult(
            text=" ".join(seg.text for seg in kept).strip(),
            language=info.language,
            language_probability=info.language_probability,
            audio_seconds=audio_seconds,
            processing_seconds=processing_seconds,
            segments=kept,
            dropped_segments=dropped,
        )

    def transcribe_file(self, path: str | Path) -> TranscriptResult:
        audio, sample_rate = load_wav(path)
        return self.transcribe(audio, sample_rate)

    @staticmethod
    def _looks_like_silence(seg: Segment) -> bool:
        """Whisper sometimes invents text (like "Thank you.") for silence or noise.

        This is the same rule Whisper itself uses: the model thinks it's probably
        not speech AND it's not confident about the words.
        """
        return (seg.no_speech_prob > config.NO_SPEECH_PROB_THRESHOLD
                and seg.avg_logprob < config.LOG_PROB_THRESHOLD)
