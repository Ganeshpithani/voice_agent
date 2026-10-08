# Local Voice Agent — Week 2: Voice-to-Text Note Taker

This is week 2 of my 4-week project to build a **voice assistant that runs fully on my own computer**. No cloud, no API keys.

In [Week 1](docs/WEEK1.md) I made sure my program could hear me and play sound back. This week, I taught it to **understand what I say**.

The mini project is a **voice note taker**:

1. I start talking.
2. It records, and **stops by itself** when I stop talking.
3. It turns my speech into text using **Whisper**, running locally.
4. It asks if I want to keep it, then saves it as a note.

I also built a small tool to **measure how accurate the speech recognition is** on my own voice, so I can pick the best Whisper model for my computer instead of guessing.

---

## Table of Contents

- [The Full 4-Week Plan](#the-full-4-week-plan)
- [What I Learned This Week](#what-i-learned-this-week)
- [How It Works (The Pipeline)](#how-it-works-the-pipeline)
- [Whisper Basics (Explained Simply)](#whisper-basics-explained-simply)
- [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
- [Audio Preprocessing](#audio-preprocessing)
- [Measuring Quality: WER and RTF](#measuring-quality-wer-and-rtf)
- [Project Structure](#project-structure)
- [What Each New File Does](#what-each-new-file-does)
- [Setup](#setup)
- [How to Use It](#how-to-use-it)
- [Testing Whisper on My Own Voice](#testing-whisper-on-my-own-voice)
- [Running the Tests](#running-the-tests)
- [Settings You Can Change](#settings-you-can-change)
- [Problems and How to Fix Them](#problems-and-how-to-fix-them)
- [Small Experiments to Try](#small-experiments-to-try)
- [What's Next (Week 3)](#whats-next-week-3)

---

## The Full 4-Week Plan

| Week | Topic | Mini Project |
|------|-------|--------------|
| 1 | Python + Audio | [Mic recorder and playback tester](docs/WEEK1.md) |
| **2** | **Speech Recognition (Whisper)** | **Voice-to-text note taker (this week)** |
| 3 | Local LLM (Ollama) + Text-to-Speech (Piper) | Ask a question in text, hear a spoken answer |
| 4 | Putting it all together | Complete local voice agent |

---

## What I Learned This Week

**Speech recognition**
- What Whisper is and how the different model sizes compare
- Why `faster-whisper` is better than the original package for running on a CPU
- Why Whisper sometimes "hears" words in silence, and how to filter them out
- How to give Whisper hints for words it gets wrong

**Audio**
- How to detect when someone starts and stops talking (VAD)
- How to convert any audio into the exact format Whisper needs
- Why you can't just drop samples to change the sample rate

**Measuring and testing**
- How to measure accuracy with **Word Error Rate (WER)**
- How to measure speed with **Real-Time Factor (RTF)**
- How to test code that needs a microphone or an AI model, **without** a microphone or a model (using fakes)

**Python**
- `dataclass` for clean result objects
- `argparse` with sub-commands for a command-line tool
- Using a `queue` to pass data safely from an audio callback to the main program
- Loading a heavy library only when it's needed (lazy loading)

---

## How It Works (The Pipeline)

A "pipeline" just means a chain of steps, where the output of one step is the input of the next:

```
Microphone
     │
     ▼
 ┌──────────────────────┐
 │ Record until silence │  audio/recorder.py + audio/vad.py
 │ (VAD decides when    │  "Is this person still talking?"
 │  to stop)            │
 └──────────────────────┘
     │  int16 audio
     ▼
 ┌──────────────────────┐
 │ Preprocess           │  audio/preprocess.py
 │ → float32, mono,     │  "Put it in the format Whisper wants"
 │   16 kHz             │
 └──────────────────────┘
     │  float32 audio
     ▼
 ┌──────────────────────┐
 │ Whisper (STT)        │  stt/whisper_stt.py
 │ → text + timing      │  "What did they say?"
 └──────────────────────┘
     │  text
     ▼
 ┌──────────────────────┐
 │ Save note            │  notes_store.py
 │ → notes/2026-09-29.md│
 └──────────────────────┘
```

STT means **Speech-To-Text**.

---

## Whisper Basics (Explained Simply)

### What is Whisper?

Whisper is a speech recognition model made by OpenAI and released for free in 2022. It was trained on about 680,000 hours of audio from the internet, which is why it handles accents and background noise quite well. It also works in many languages.

The best part for this project: **the model files can be downloaded and run completely offline**.

### How it works (the short version)

1. Whisper splits audio into pieces of up to 30 seconds.
2. It turns each piece into a **spectrogram**: a picture showing which sound frequencies are present at each moment.
3. The model "reads" this picture and writes out the most likely words, one small piece at a time.

### Model sizes

Whisper comes in different sizes. Bigger models are more accurate but slower and need more memory.

| Model | Download size (about) | Speed | Accuracy |
|-------|----------------------|-------|----------|
| `tiny` | 75 MB | Fastest | Lowest |
| `base` | 145 MB | Fast | OK |
| `small` | 480 MB | Medium | Good |
| `medium` | 1.5 GB | Slow on CPU | Very good |
| `large-v3` | 3 GB | Very slow on CPU | Best |

Models ending in **`.en`** (like `base.en`) only understand English. For English speech, they are a bit more accurate than the same-size multilingual model. I use `base.en` as the default because it's a good balance on a normal laptop.

**The sizes are only a rough guide.** The real speed depends on your computer, which is why I built the evaluation tool (see [Testing Whisper on My Own Voice](#testing-whisper-on-my-own-voice)).

### Why faster-whisper?

There are two ways to run Whisper in Python:

- **`openai-whisper`**: the original package. It uses PyTorch and is quite slow on a CPU.
- **`faster-whisper`**: runs the same models using a faster engine called CTranslate2. It gives the same results, but is several times faster and uses less memory.

Since everything runs on my own computer, speed matters a lot. So I use `faster-whisper`.

### int8 — making the model smaller and faster

A model is basically millions of numbers. Normally each number is stored with high detail (32-bit or 16-bit). **int8** stores each number with less detail (8-bit). The model becomes faster and smaller, and the accuracy loss is usually very small. This is called **quantization**. On a CPU, I use `int8`.

### Beam size

When Whisper picks words, it can either:
- take the single best guess at each step (`beam_size = 1`, fastest), or
- keep several possible sentences at once and choose the best one at the end (`beam_size = 5`, a bit slower but a bit more accurate).

This is called **beam search**.

### Hallucinations — when Whisper makes things up

This surprised me. If you give Whisper silence or just noise, it sometimes still writes text, like **"Thank you."** or **"Thanks for watching!"** This happens because it learned from lots of online videos that end with those words.

I use three protections against this:

1. **VAD filter** (`vad_filter=True`): faster-whisper skips silent parts before transcribing.
2. **Segment filter**: Whisper gives each piece of text two scores:
   - `no_speech_prob`: how likely it thinks this part is **not** speech
   - `avg_logprob`: how confident it is about the words (closer to 0 = more confident)

   If a piece is probably not speech **and** the model isn't confident, I drop it. This is the same rule Whisper uses internally.
3. **`condition_on_previous_text=False`**: this stops Whisper from using its earlier text as context. It helps prevent the same sentence from repeating over and over.

### Initial prompt — giving Whisper hints

Whisper often gets unusual words wrong, like names or tech terms ("Ollama" may come out as "Oh Lama"). You can give it a hint with `WHISPER_INITIAL_PROMPT` in `config.py`:

```python
WHISPER_INITIAL_PROMPT = "Ollama, Piper, Whisper, Python"
```

Whisper then treats these words as more likely.

---

## Voice Activity Detection (VAD)

In Week 1, I recorded for a fixed time, like 5 seconds. That's not how a real assistant works. You don't want to wait 5 seconds after saying "hi", and a long sentence would get cut off.

**VAD** means **Voice Activity Detection**: working out when someone is talking and when they stopped.

### How my VAD works

It's a simple **energy-based** VAD. "Energy" just means loudness.

1. **Calibrate:** for the first half second, it listens to the room and measures the background noise. (That's why the app asks you to stay quiet for a moment.)
2. **Set a threshold:** anything **10 dB louder** than the room noise counts as speech. The threshold also has limits (-50 to -25 dBFS), so it still works in very quiet or noisy rooms.
3. **Check small chunks:** audio is checked in 30-millisecond pieces.

### The three states

```
  WAITING ───(3 loud chunks in a row)───► SPEAKING ───(1.2 s of silence)───► DONE
     │                                        │
     └──(no speech for 8 s)──► DONE           └──(60 s limit)──► DONE
         reason: "no_speech"                      reason: "max_length"
```

A few details that made a big difference:

- **3 chunks in a row:** a single click or desk tap is only one loud chunk, so it doesn't count as speech.
- **Pre-roll:** it keeps the last 0.3 seconds of audio from **before** speech was detected. Without this, the start of the first word gets cut off.
- **Trimming the tail:** after you stop, it waits 1.2 seconds to be sure you're done, then removes most of that silence from the recording.

### Why the logic is in its own class

The `SpeechSegmenter` class doesn't know anything about microphones. You just feed it chunks of audio, and it tells you what's happening. This means I can **test it with fake audio** (a beep with silence around it) and check it makes the right decisions. No microphone needed.

### Limits

A loud fan, music, or a TV can fool an energy-based VAD, because it only looks at loudness. A smarter option is a small neural network VAD, like **Silero VAD**. I may switch to it later.

### The queue

The microphone stream calls my callback function many times per second. The callback has to be very fast, or audio gets dropped. So it only does one thing: put each chunk into a **queue**. The main program takes chunks out of the queue and runs the VAD. A queue is a safe way to pass data between two parts of a program running at the same time.

---

## Audio Preprocessing

Whisper needs audio in exactly this format:

| Property | What Whisper wants | What I might have |
|----------|--------------------|-------------------|
| Number type | `float32` from -1.0 to 1.0 | `int16` from -32768 to 32767 |
| Channels | Mono (1 channel) | Maybe stereo from a WAV file |
| Sample rate | 16,000 Hz | Maybe 44,100 or 48,000 Hz from a WAV file |

`prepare_for_whisper()` in `audio/preprocess.py` fixes all three:

1. **`to_float32`:** divides by 32768 to get numbers between -1.0 and 1.0.
2. **`to_mono`:** if there are two channels, it averages them into one.
3. **`resample`:** changes the sample rate.

### Why resampling needs care

To go from 48,000 Hz to 16,000 Hz, you might think "just keep every 3rd sample". But that creates strange noise, called **aliasing**. High sounds that don't fit in the new rate get "folded" into fake lower sounds. The proper way is to filter out those high sounds first, then reduce the samples. `scipy.signal.resample_poly` does both.

There are also two optional steps:

- **`trim_silence`:** cuts quiet parts from the start and end.
- **`normalize_peak`:** makes quiet recordings louder. It skips audio that is basically silence, because boosting silence would just boost the noise.

---

## Measuring Quality: WER and RTF

"It seems to work" isn't a good test. I wanted real numbers.

### WER — Word Error Rate (accuracy)

WER counts how many words are wrong, compared to what was really said:

```
WER = (substituted words + deleted words + inserted words) / words in the real sentence
```

Example:

```
What I said:    "turn on the  kitchen light"
Whisper wrote:  "turn on a    kitchen light please"
                          ↑                   ↑
                   substitution          insertion
```

2 errors ÷ 5 words = **40% WER**.

- **0%** = perfect
- **Lower is better**
- It can go above 100% if Whisper adds lots of extra words

Before comparing, both texts are **normalized**: made lowercase and with punctuation removed. So "Hello, world!" and "hello world" count as the same.

One limit: numbers are not converted. "5" and "five" count as different words. So a sentence with numbers may show errors even when Whisper understood it correctly.

To count errors, I use **edit distance** (also called Levenshtein distance). It finds the smallest number of changes needed to turn one sentence into the other.

### RTF — Real-Time Factor (speed)

```
RTF = time taken to transcribe / length of the audio
```

- **RTF 0.2** = 10 seconds of audio took 2 seconds. Fast.
- **RTF 1.0** = it takes as long as the audio itself.
- **Above 1.0** = slower than real time. Too slow for a voice assistant.

For a voice assistant, low RTF matters as much as low WER. Nobody wants to wait 10 seconds for a reply.

---

## Project Structure

Files marked **NEW** were added this week.

```
voice_agent/
│
├── note_taker.py        # NEW - Week 2 mini project
├── evaluate_stt.py      # NEW - measure Whisper accuracy and speed
├── notes_store.py       # NEW - save/read notes as Markdown
├── mic_tester.py        # Week 1 mini project
├── config.py            # All settings (Week 2 settings added)
├── requirements.txt
├── .gitignore
├── README.md
│
├── docs/
│   └── WEEK1.md         # Week 1 write-up
│
├── audio/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── devices.py
│   ├── recorder.py      # + record_until_silence()
│   ├── player.py
│   ├── wav_io.py
│   ├── levels.py        # now also works with float audio
│   ├── preprocess.py    # NEW - get audio ready for Whisper
│   └── vad.py           # NEW - detect when someone is talking
│
├── stt/                 # NEW - speech-to-text package
│   ├── __init__.py
│   ├── exceptions.py    # STTError
│   ├── whisper_stt.py   # Whisper wrapper
│   └── metrics.py       # Word Error Rate
│
├── tests/
│   ├── test_audio.py    # Week 1 tests
│   └── test_stt.py      # NEW - Week 2 tests
│
├── models/              # Whisper model is downloaded here (not uploaded to Git)
├── notes/               # Your saved notes (not uploaded to Git)
└── temp/                # Recordings (not uploaded to Git)
```

---

## What Each New File Does

### `note_taker.py` — the main program

Loads the Whisper model once at the start (this takes a few seconds), then shows a menu:

```
=== Voice Note Taker ===
 1. Record a note (stops when you stop talking)
 2. Transcribe a WAV file
 3. Show today's notes
 4. Show settings
 0. Quit
```

After each recording, it shows the text plus some stats: the language, audio length, how long it took, and the RTF. Then it asks if you want to save the note.

### `stt/whisper_stt.py` — the Whisper wrapper

The `WhisperSTT` class:

- **`load()`** loads the model. The first time, it downloads it into `models/whisper/`. It only loads once, and after that it's reused.
- **`transcribe(audio, sample_rate)`** takes any audio, preprocesses it, runs Whisper, removes likely hallucinations, and returns a `TranscriptResult`.
- **`transcribe_file(path)`** does the same for a WAV file.

`TranscriptResult` is a `dataclass` that holds the text, language, timings, and each segment. It also works out the RTF.

Two design choices I'm happy with:

- **Lazy import:** `faster_whisper` is only imported inside `load()`. If it's not installed, you get a clear message telling you what to install, instead of a crash when the program starts.
- **You can pass in a fake model:** `WhisperSTT(model=fake)`. The tests use this to check my code without downloading a real model.

### `stt/metrics.py` — measuring accuracy

- `normalize_text()` makes text lowercase and removes punctuation (but keeps words like "don't").
- `word_error_rate()` compares two sentences and returns the WER, plus how many substitutions, deletions, and insertions there were.

### `stt/exceptions.py`

One error type, `STTError`, for when the model can't load or transcription fails.

### `audio/vad.py` — voice activity detection

- `EnergyVAD` decides if a single chunk is speech.
- `SpeechSegmenter` decides when speech **starts** and **ends**, using the states described above.

### `audio/recorder.py` — new function

`record_until_silence()` opens the mic stream, sends chunks through the queue to the `SpeechSegmenter`, and returns the audio and the reason it stopped. It also calls `on_event` when it's ready to listen and when it hears you, so the app can show messages like "Listening... speak now."

### `audio/preprocess.py` — preprocessing

All the conversion steps described in [Audio Preprocessing](#audio-preprocessing).

### `notes_store.py` — saving notes

Saves one Markdown file per day. A note file looks like this:

```markdown
# Notes — 2026-09-29

- **14:30:15** — Buy milk and eggs on the way home
- **16:02:41** — Idea for week 3: make the agent answer in a short sentence first
```

Markdown files are easy to read on GitHub or in any text editor.

### `evaluate_stt.py` — testing accuracy on my voice

Explained in [Testing Whisper on My Own Voice](#testing-whisper-on-my-own-voice).

---

## Setup

If you already set up Week 1, just install the new packages:

```bash
pip install -r requirements.txt
```

New packages this week:

- **faster-whisper** — runs Whisper models quickly
- **scipy** — used for resampling audio properly

### Fresh setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**The first run downloads the Whisper model**, so you need internet the first time. After that, it works fully offline.

---

## How to Use It

```bash
python note_taker.py
```

1. Wait for "Model ready".
2. Choose **option 1**.
3. **Stay quiet** for half a second while it measures the room noise.
4. When you see `>> Listening... speak now.`, say your note.
5. Stop talking. After about a second, it stops recording by itself.
6. Check the text, then press **Enter** to save it (or type `n` to skip).

Choose **option 3** to see today's notes.

**Option 2** lets you transcribe any 16-bit WAV file. For example, a recording you made with `mic_tester.py` in Week 1.

---

## Testing Whisper on My Own Voice

Public test results don't tell me how Whisper does with **my** voice, **my** accent, **my** mic, and **my** laptop. So I built `evaluate_stt.py`.

### Step 1 — Record test sentences

```bash
python evaluate_stt.py record
```

It shows sentences one by one. I read each one out loud. The sentences include easy ones, numbers, names, and tech words, so I can see where Whisper struggles. Each recording is saved as a pair:

```
tests/stt_samples/sample_01.wav   ← my voice
tests/stt_samples/sample_01.txt   ← what I actually said
```

### Step 2 — Compare models

```bash
python evaluate_stt.py run --models tiny.en base.en small.en
```

For each model, it transcribes every sample and shows the WER and RTF. When something is wrong, it shows what I said next to what Whisper wrote. At the end, there's a summary table like this:

```
=== Summary ===
Model            WER    RTF    Load
tiny.en        ...%   ...    ...s
base.en        ...%   ...    ...s
small.en       ...%   ...    ...s
```

The overall WER counts **all errors divided by all words**. So longer sentences count more than short ones.

### My results

*(Fill in after running on your own computer.)*

| Model | WER | RTF | Load time |
|-------|-----|-----|-----------|
| `tiny.en` |39.3%  | 0.08| 11.1s |
| `base.en` |34.4%  | 0.13| 0.7s |
| `small.en` |34.4%  | 0.36| 48.2s|

My computer: GPU: GTX 1650, 4GB RAM.

---

## Running the Tests

```bash
python -m pytest
```

You should see:

```
25 passed
```

None of the tests need a microphone or a downloaded model. They check:

- **Preprocessing:** number conversion, stereo to mono, resampling length, trimming, and normalizing.
- **VAD:** it stops after silence, it gives up when nobody talks, it stops at the max length, and a single click doesn't count as speech.
- **WER:** it gets the exact error counts right for known examples.
- **Whisper wrapper:** using a **fake model** that returns one real sentence and one "Thank you." that looks like silence. The test checks that the fake one is removed, and the audio was converted to 16 kHz float32 before being sent to the model.
- **Recording loop:** using a **fake microphone** that sends pre-made audio (noise, then a beep, then silence) to the real `record_until_silence()` function.
- **Notes:** saving, reading, and rejecting empty notes.

Writing fakes was one of the most useful things I learned this week. It lets me test code that normally needs hardware or a big AI model.

---

## Settings You Can Change

All in `config.py`.

### Whisper

| Setting | Default | What it does |
|---------|---------|--------------|
| `WHISPER_MODEL` | `"base.en"` | Model size (see table above) |
| `WHISPER_DEVICE` | `"cpu"` | `"cpu"`, `"cuda"` (NVIDIA GPU), or `"auto"` |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | `"int8"` for CPU, `"float16"` for GPU |
| `WHISPER_LANGUAGE` | `"en"` | `None` = auto-detect (needs a model without `.en`) |
| `WHISPER_BEAM_SIZE` | `5` | `1` = fastest, `5` = a bit more accurate |
| `WHISPER_VAD_FILTER` | `True` | Skip silent parts inside Whisper |
| `WHISPER_INITIAL_PROMPT` | `None` | Hint words, like `"Ollama, Piper"` |
| `NO_SPEECH_PROB_THRESHOLD` | `0.6` | Used to drop segments that look like silence |
| `LOG_PROB_THRESHOLD` | `-1.0` | Used to drop segments with low confidence |

### Recording and VAD

| Setting | Default | What it does |
|---------|---------|--------------|
| `SILENCE_SECONDS` | `1.2` | Stop after this much silence |
| `START_TIMEOUT_SECONDS` | `8.0` | Give up if you don't start talking in this time |
| `MAX_RECORD_SECONDS` | `60.0` | Longest possible recording |
| `VAD_CALIBRATION_SECONDS` | `0.5` | Time spent measuring room noise |
| `VAD_MARGIN_DB` | `10.0` | How much louder than the room speech must be |
| `VAD_MIN_THRESHOLD_DBFS` | `-50.0` | Lowest allowed threshold |
| `VAD_MAX_THRESHOLD_DBFS` | `-25.0` | Highest allowed threshold |
| `PRE_ROLL_SECONDS` | `0.3` | Audio kept from before speech started |

### Notes

| Setting | Default | What it does |
|---------|---------|--------------|
| `KEEP_NOTE_AUDIO` | `True` | Also save each note's audio in `temp/` |

---

## Problems and How to Fix Them

**"Could not load Whisper model"**
- The first run needs internet to download the model.
- Check the model name is spelled right (for example `base.en`, not `base-en`).
- If you set `WHISPER_DEVICE = "cuda"` but don't have an NVIDIA GPU set up, change it back to `"cpu"`.

**It stops recording while I'm still talking**
Increase `SILENCE_SECONDS` to something like `1.8`. Short pauses between words can count as "silence".

**It never stops recording**
The room is probably too noisy, so your "silence" is still louder than the threshold. Try:
- staying fully quiet during the calibration at the start
- **increasing** `VAD_MARGIN_DB`, for example to `15`
- in a very noisy room, **raising** `VAD_MAX_THRESHOLD_DBFS`, for example to `-20`, so the threshold is allowed to go higher
- moving away from fans or other noise

**"I didn't hear anything"**
Your voice may be too quiet for the VAD. Check your mic with `python mic_tester.py` (option 3). You can also **lower** `VAD_MARGIN_DB`, for example to `6`.

**Whisper writes "Thank you." when I said nothing**
This is a hallucination (see above). Make sure `WHISPER_VAD_FILTER = True`. You can also make the filter stricter by lowering `NO_SPEECH_PROB_THRESHOLD` a little, like to `0.5`.

**It's too slow**
- Try a smaller model like `tiny.en`.
- Set `WHISPER_BEAM_SIZE = 1`.
- If you have an NVIDIA GPU, set `WHISPER_DEVICE = "cuda"` and `WHISPER_COMPUTE_TYPE = "float16"`.

**Names or tech words come out wrong**
Add them to `WHISPER_INITIAL_PROMPT`.

**The first transcription is slower than the rest**
That's normal. The first run "warms up" the model.

---

## Small Experiments to Try

- Compare `tiny.en`, `base.en`, and `small.en` with `evaluate_stt.py`. Is the accuracy gain worth the slower speed?
- Run the evaluation with `WHISPER_BEAM_SIZE = 1`, then `5`. How much do WER and RTF change?
- Look at which words fail. Add them to `WHISPER_INITIAL_PROMPT` and run the evaluation again. Did the WER go down?
- Set `WHISPER_VAD_FILTER = False`, then record a note where you say nothing. Does Whisper make something up?
- Set `WHISPER_LANGUAGE = None` and `WHISPER_MODEL = "base"`, then speak in another language you know. Does it detect it correctly?
- Record in a quiet room, then with a fan or music on. Compare the WER.

---

## What's Next (Week 3)

Now the agent can **hear** and **understand** me. Next week, I'll give it a **brain** and a **voice**:

- **Ollama:** runs a large language model (LLM) locally, so the agent can think of an answer
- **Prompting and conversation context:** so it remembers what we talked about earlier in the chat
- **Piper TTS:** turns the answer into spoken audio. TTS means Text-To-Speech.

The mini project: type a question, the local LLM answers, and Piper speaks the answer out loud. In Week 4, I'll connect this week's speech-to-text to the front of it, and the full voice agent will be complete.
