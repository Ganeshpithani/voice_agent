"""Project settings. Change values here instead of inside the code."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"
NOTES_DIR = BASE_DIR / "notes"
STT_SAMPLES_DIR = BASE_DIR / "tests" / "stt_samples"

# ============================================================
# Week 1 — Audio format
# ============================================================
# Whisper expects 16 kHz mono audio, so we record in that format from day one.
SAMPLE_RATE = 16000   # samples per second (Hz)
CHANNELS = 1          # 1 = mono, 2 = stereo
DTYPE = "int16"       # 16-bit samples, the standard for WAV files

RECORD_SECONDS = 5    # default length for fixed-time recording

# None = system default. Or a device number (e.g. 3) or part of a name (e.g. "USB").
INPUT_DEVICE = None
OUTPUT_DEVICE = None

TOO_QUIET_DBFS = -40.0   # below this, recordings get a "too quiet" warning

# ============================================================
# Week 2 — Voice activity detection (record until you stop talking)
# ============================================================
VAD_FRAME_MS = 30              # audio is checked in small chunks of this length
VAD_CALIBRATION_SECONDS = 0.5  # stay quiet this long at the start so it can measure room noise
VAD_MARGIN_DB = 10.0           # speech must be this much louder than room noise
VAD_MIN_THRESHOLD_DBFS = -50.0 # threshold never goes lower than this (very quiet room)
VAD_MAX_THRESHOLD_DBFS = -25.0 # threshold never goes higher than this (noisy room)
SILENCE_SECONDS = 1.2          # stop after this much silence once you have started talking
START_TIMEOUT_SECONDS = 8.0    # give up if you don't start talking within this time
MAX_RECORD_SECONDS = 60.0      # hard limit for one recording
PRE_ROLL_SECONDS = 0.3         # keep a little audio from just before speech started

# ============================================================
# Week 2 — Whisper speech-to-text (faster-whisper)
# ============================================================
# Model sizes: "tiny", "base", "small", "medium", "large-v3"
# Also English-only versions: "tiny.en", "base.en", "small.en", "medium.en"
# Bigger = more accurate but slower and more memory.
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cpu"          # "cpu", "cuda" (NVIDIA GPU) or "auto"
WHISPER_COMPUTE_TYPE = "int8"   # "int8" is fast on CPU. For GPU use "float16".
WHISPER_LANGUAGE = "en"         # None = detect automatically (needs a non-".en" model)
WHISPER_BEAM_SIZE = 5           # higher = a bit more accurate, a bit slower. 1 = fastest.
WHISPER_VAD_FILTER = True       # let faster-whisper skip silent parts (reduces made-up text)
# Words Whisper often gets wrong can be given as a hint, e.g. "Ollama, Piper, Python".
WHISPER_INITIAL_PROMPT = None

# Segments that look like silence are dropped (Whisper sometimes "hears" words in silence).
NO_SPEECH_PROB_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0

# ============================================================
# Week 2 — Note taker
# ============================================================
KEEP_NOTE_AUDIO = True   # also save each note's audio as a WAV in temp/
