"""Week 2 tests. None of these need a microphone or a downloaded model."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from audio.player import make_tone
from audio.preprocess import (normalize_peak, prepare_for_whisper, resample,
                              to_float32, to_mono, trim_silence)
from audio.vad import DONE, SPEAKING, EnergyVAD, SpeechSegmenter
from notes_store import read_notes, save_note
from stt.metrics import normalize_text, word_error_rate
from stt.whisper_stt import WhisperSTT

SR = 16000
FRAME = 480  # 30 ms at 16 kHz


# ---------- preprocessing ----------

def test_to_float32_range():
    audio = np.array([-32768, 0, 16384], dtype=np.int16)
    out = to_float32(audio)
    assert out.dtype == np.float32
    assert out[0] == -1.0 and out[1] == 0.0 and out[2] == pytest.approx(0.5)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [0.5, 0.5])


def test_resample_48k_to_16k_length():
    one_second = np.zeros(48000, dtype=np.float32)
    assert len(resample(one_second, 48000, 16000)) == 16000


def test_prepare_for_whisper_output_format():
    stereo_48k = np.zeros((48000, 2), dtype=np.int16)
    out = prepare_for_whisper(stereo_48k, 48000)
    assert out.dtype == np.float32 and out.ndim == 1 and len(out) == 16000


def test_trim_silence_keeps_speech_only():
    silence = np.zeros(SR, dtype=np.float32)
    tone = to_float32(make_tone(seconds=0.5, sample_rate=SR))
    trimmed = trim_silence(np.concatenate([silence, tone, silence]), SR, pad_seconds=0.0)
    assert 0.4 < len(trimmed) / SR < 0.6


def test_normalize_peak_and_silence():
    quiet = to_float32(make_tone(volume=0.05))
    assert np.max(np.abs(normalize_peak(quiet, -3.0))) == pytest.approx(10 ** (-3 / 20), rel=1e-3)
    silence = np.zeros(100, dtype=np.float32)
    assert np.array_equal(normalize_peak(silence), silence)


# ---------- VAD / segmenter ----------

def frames_of(audio):
    return [audio[i:i + FRAME] for i in range(0, len(audio) - FRAME + 1, FRAME)]


def make_segmenter(**kwargs):
    vad = EnergyVAD(calibration_frames=15, margin_db=10.0)
    return SpeechSegmenter(vad, SR, FRAME, **kwargs)


def room_noise(seconds, level=30):
    rng = np.random.default_rng(0)
    return rng.integers(-level, level, int(seconds * SR)).astype(np.int16)


def test_segmenter_stops_after_silence():
    seg = make_segmenter(silence_seconds=1.0)
    audio = np.concatenate([room_noise(0.6), make_tone(seconds=1.0), room_noise(2.0)])
    events = [seg.feed(f) for f in frames_of(audio)]
    assert "calibrated" in events and "speech_start" in events
    assert seg.state == DONE and seg.stop_reason == "silence"
    length = len(seg.result()) / SR
    assert 1.0 < length < 1.8  # speech + pre-roll + a bit of tail, not the whole 2 s of silence


def test_segmenter_no_speech_timeout():
    seg = make_segmenter(start_timeout=1.0)
    for f in frames_of(room_noise(2.0)):
        seg.feed(f)
    assert seg.stop_reason == "no_speech" and seg.result() is None


def test_segmenter_max_length():
    seg = make_segmenter(max_seconds=2.0)
    audio = np.concatenate([room_noise(0.6), make_tone(seconds=5.0)])
    for f in frames_of(audio):
        seg.feed(f)
        if seg.done:
            break
    assert seg.stop_reason == "max_length"


def test_short_click_is_not_speech():
    seg = make_segmenter(start_timeout=2.0)
    click = make_tone(seconds=0.03)  # one frame only
    audio = np.concatenate([room_noise(0.6), click, room_noise(0.3)])
    for f in frames_of(audio):
        seg.feed(f)
    assert seg.state != SPEAKING


# ---------- WER ----------

def test_normalize_text():
    assert normalize_text("Hello, World! Don't stop.") == "hello world don't stop"


def test_wer_example_from_docs():
    r = word_error_rate("turn on the kitchen light", "turn on a kitchen light please")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 1)
    assert r.wer == pytest.approx(0.4)


def test_wer_perfect_and_deletion():
    assert word_error_rate("Hello there.", "hello there").wer == 0.0
    r = word_error_rate("one two three four", "one three four")
    assert r.deletions == 1 and r.wer == pytest.approx(0.25)


# ---------- WhisperSTT with a fake model ----------

class FakeModel:
    """Pretends to be faster-whisper so we can test our code without a download."""

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        self.audio = audio
        segments = [
            SimpleNamespace(start=0.0, end=1.0, text=" Hello world.", avg_logprob=-0.2, no_speech_prob=0.01),
            SimpleNamespace(start=1.0, end=2.0, text=" Thank you.", avg_logprob=-1.5, no_speech_prob=0.9),
        ]
        info = SimpleNamespace(language="en", language_probability=0.99)
        return iter(segments), info


def test_whisper_stt_with_fake_model():
    fake = FakeModel()
    stt = WhisperSTT(model=fake)
    result = stt.transcribe(make_tone(seconds=2.0, sample_rate=48000), 48000)
    assert result.text == "Hello world."        # the "silence" segment was dropped
    assert result.dropped_segments == 1
    assert fake.audio.dtype == np.float32 and len(fake.audio) == 32000  # resampled to 16 kHz
    assert result.audio_seconds == pytest.approx(2.0)
    assert fake.kwargs["condition_on_previous_text"] is False


def test_whisper_stt_too_short_audio():
    stt = WhisperSTT(model=FakeModel())
    assert stt.transcribe(np.zeros(100, dtype=np.int16), SR).text == ""


# ---------- notes ----------

def test_save_and_read_notes(tmp_path):
    when = datetime(2026, 9, 29, 14, 30, 15)
    save_note("buy   milk", when=when, notes_dir=tmp_path)
    save_note("call mom", when=when, notes_dir=tmp_path)
    text = read_notes(when.date(), notes_dir=tmp_path)
    assert text.startswith("# Notes — 2026-09-29")
    assert "- **14:30:15** — buy milk" in text and text.count("- **") == 2


def test_empty_note_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_note("   ", notes_dir=tmp_path)


# ---------- record_until_silence with a fake microphone ----------

class FakeStream:
    """Acts like sounddevice.InputStream: sends pre-made audio to the callback."""

    def __init__(self, audio, blocksize, callback, **kwargs):
        self.audio, self.blocksize, self.callback = audio, blocksize, callback

    def __enter__(self):
        for i in range(0, len(self.audio) - self.blocksize + 1, self.blocksize):
            block = self.audio[i:i + self.blocksize].reshape(-1, 1)
            self.callback(block, self.blocksize, None, None)
        return self

    def __exit__(self, *exc):
        return False


def test_record_until_silence_with_fake_mic(monkeypatch):
    import audio.recorder as recorder

    audio = np.concatenate([room_noise(0.6), make_tone(seconds=1.0), room_noise(2.0)])
    fake_sd = SimpleNamespace(
        InputStream=lambda **kw: FakeStream(audio, kw["blocksize"], kw["callback"]),
        PortAudioError=RuntimeError,
    )
    monkeypatch.setattr(recorder, "require_sounddevice", lambda: fake_sd)

    events = []
    result, reason = recorder.record_until_silence(on_event=events.append)
    assert reason == "silence"
    assert events == ["calibrated", "speech_start"]
    assert result.dtype == np.int16 and 1.0 < len(result) / SR < 1.8
