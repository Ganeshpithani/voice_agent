# Local Voice Agent — Week 1: Microphone Recorder & Playback Tester

This is the first week of my 4-week project to build a **voice assistant that runs fully on my own computer**. No cloud, no API keys. You talk, it listens, it thinks with a local AI model, and it talks back.

Before I can build any of that, I need to be sure one basic thing works: **can my program hear me, and can it play sound back?** That is what Week 1 is about.

So this week I built a small command-line tool that can:

- list all the microphones and speakers on my computer
- play a test beep to check my speakers
- show a live "volume bar" that moves when I talk, to check my mic
- record my voice and save it as a WAV file
- play the recording back
- show details about the file (length, sample rate, size, and so on)
- warn me if my recording is too quiet or too loud

It sounds simple, but I learned a lot about how digital audio actually works while building it.

---

## Table of Contents

- [The Full 4-Week Plan](#the-full-4-week-plan)
- [What I Learned This Week](#what-i-learned-this-week)
- [Audio Basics (Explained Simply)](#audio-basics-explained-simply)
- [Project Structure](#project-structure)
- [What Each File Does](#what-each-file-does)
- [Setup](#setup)
- [How to Use It](#how-to-use-it)
- [Running the Tests](#running-the-tests)
- [Settings You Can Change](#settings-you-can-change)
- [Problems I Hit and How to Fix Them](#problems-i-hit-and-how-to-fix-them)
- [Small Experiments to Try](#small-experiments-to-try)
- [What's Next (Week 2)](#whats-next-week-2)

---

## The Full 4-Week Plan

| Week | Topic | Mini Project |
|------|-------|--------------|
| **1** | **Python + Audio** | **Mic recorder and playback tester (this repo)** |
| 2 | Speech Recognition (Whisper) | Voice-to-text note taker |
| 3 | Local LLM (Ollama) + Text-to-Speech (Piper) | Ask a question in text, hear a spoken answer |
| 4 | Putting it all together | Complete local voice agent |

Each week builds on the one before. The `audio/` code I wrote this week will be reused in every later week.

---

## What I Learned This Week

**Python**
- Splitting code into **modules** (separate files) where each file has one job
- Writing small, clear **functions** with default values and type hints
- Making my own **custom exceptions** so errors are easy to understand and catch
- Using **pip** and a **virtual environment** so the project's packages don't mix with the rest of my system
- Writing **tests** with `pytest`

**Audio**
- What **sample rate**, **channels**, and **bit depth** mean
- How a **WAV file** is built
- How to measure loudness in **dBFS**
- What **clipping** is and why it ruins recordings
- The difference between **recording a fixed length** and **streaming audio live** with a callback

---

## Audio Basics (Explained Simply)

I didn't know most of this before starting, so I'm writing it down here in plain words.

### Sound on a computer is just a list of numbers

A microphone turns air vibrations into an electrical signal. The computer measures that signal many times per second and stores each measurement as a number. Each number is called a **sample**.

So a recording is really just a long list of numbers. In Python, I store it as a **NumPy array**.

### Sample rate — how many numbers per second

The **sample rate** is how many samples are taken each second. It's measured in **Hz**.

| Sample rate | Where it's used |
|-------------|-----------------|
| 8,000 Hz | Old phone calls |
| **16,000 Hz** | **Speech recognition (what I use)** |
| 44,100 Hz | Music CDs |
| 48,000 Hz | Video and most modern devices |

A higher rate captures more detail, but also makes bigger files. For speech, 16,000 Hz is enough. And the main reason I picked it: **Whisper (the speech-to-text model for Week 2) expects 16 kHz audio**. By recording in that format from day one, I won't need to convert anything later.

### Channels — mono or stereo

- **Mono (1 channel):** one stream of sound
- **Stereo (2 channels):** separate left and right streams

A voice assistant only needs to hear one voice, so I use **mono**. It's also half the file size.

### Bit depth — how exact each number is

**Bit depth** is how much space each sample gets. I use **16-bit**, which means each sample is a whole number between **-32,768 and 32,767**. This is the standard for WAV files and speech. In the code, this is the `int16` data type.

### WAV files

A **WAV** file is the simplest audio format. It has a small **header** at the start (which stores the sample rate, channels, and bit depth), followed by the raw sample numbers. There's no compression, so it's easy to read and write. Python even has a built-in `wave` module for it, so I didn't need an extra library.

**Quick math for file size:**
16,000 samples/second × 2 bytes per sample × 1 channel = **32 KB per second**.
So a 5-second recording is about **156 KB**.

### dBFS — measuring loudness

**dBFS** means "decibels relative to full scale". It sounds scary, but the idea is simple:

- **0 dBFS** = the loudest sound that can be stored
- Every number below that is quieter, so values are **negative**
- **-inf** (minus infinity) = total silence

Rough guide for speech near a mic:

| Level | Meaning |
|-------|---------|
| 0 dBFS | Too loud, probably clipping |
| -10 to -30 dBFS | Good speaking level |
| below -40 dBFS | Too quiet, the mic might be wrong or muted |

My program shows two numbers:
- **Peak** — the single loudest moment
- **Average (RMS)** — the overall loudness. RMS is just a math way of taking the average that works well for sound waves

### Clipping — when it's too loud

If the sound is louder than the biggest number 16-bit can hold (32,767), the extra part is simply cut off. This is called **clipping**, and it makes the audio sound crackly and broken. It also makes speech recognition worse. My tester counts clipped samples and warns you if it finds any.

### Two ways to record

1. **Fixed recording** (`record()` in `recorder.py`): "Record exactly 5 seconds, then give me all the audio." Simple, and good for testing.
2. **Streaming with a callback** (`monitor_levels()` in `recorder.py`): the audio library keeps calling my function many times per second, each time with a small new chunk of sound. This is how the live volume bar works.

The streaming way is the one a real voice agent uses, because it needs to listen all the time. I'll come back to it in Week 4.

---

## Project Structure

```
voice_agent/
│
├── mic_tester.py        # The main program (the menu you interact with)
├── config.py            # All settings in one place
├── requirements.txt     # Python packages needed
├── .gitignore           # Files Git should not upload
├── README.md            # This file
│
├── audio/               # All the audio code (reused in later weeks)
│   ├── __init__.py
│   ├── exceptions.py    # Custom error types
│   ├── devices.py       # Finds mics and speakers
│   ├── recorder.py      # Records from the mic
│   ├── player.py        # Plays sound + makes a test beep
│   ├── wav_io.py        # Saves and loads WAV files
│   └── levels.py        # Measures loudness and clipping
│
├── tests/
│   ├── __init__.py
│   └── test_audio.py    # Tests that run without a microphone
│
└── temp/                # Recordings are saved here (not uploaded to Git)
```

The `__init__.py` files tell Python that a folder is a **package**, so I can write imports like `from audio.recorder import record`.

---

## What Each File Does

### `mic_tester.py` — the main program

This shows the menu, reads your choice, and calls the right function. It also catches errors. If something goes wrong (for example, no mic is found, or you type letters instead of a number), it prints a clear message and returns to the menu instead of crashing.

If you press **Ctrl+C** while recording or playing, it stops the audio and goes back to the menu.

### `config.py` — settings

All the numbers that I might want to change are here: sample rate, channels, recording length, which device to use, and so on. This way, I never have to dig through the code to change a setting.

### `audio/exceptions.py` — custom errors

I made three error types:

```
AudioError          ← the parent. Catch this to catch any audio problem
├── DeviceError     ← mic or speaker missing, busy, or doesn't support the settings
└── WavFileError    ← WAV file missing, broken, or wrong format
```

Because they all come from `AudioError`, the main program only needs one line, `except AudioError:`, to handle all of them. I also use `raise ... from exc`, which keeps the original error attached. That makes debugging much easier.

### `audio/devices.py` — finding devices

- `print_devices()` prints a table of every mic and speaker, with a number for each one. `>` marks the default mic and `<` marks the default speaker.
- `require_sounddevice()` handles a common setup problem. The `sounddevice` library needs a system library called **PortAudio**. If PortAudio is missing, instead of a confusing crash, you get a message telling you exactly how to install it for your operating system.

### `audio/recorder.py` — recording

- `record(seconds)` records a fixed amount of time and returns the audio as a NumPy array. Before recording, it checks that the mic supports the chosen settings, so you get a clear error early.
- `monitor_levels(seconds)` opens a live stream and shows a moving volume bar, like this:

```
[####################....................]  -28.4 dBFS
```

One thing I learned: the callback function must be **very fast**. If it's slow, audio chunks get dropped. So the callback only saves the latest level, and the main loop does the printing.

### `audio/player.py` — playback

- `play(audio, sample_rate)` plays a NumPy array through the speakers.
- `play_wav(path)` loads a WAV file and plays it.
- `make_tone()` creates a 440 Hz beep using a sine wave. This lets me test my speakers even without a microphone. It also fades the beep in and out very quickly, because a sudden start or stop makes a "click" sound.

### `audio/wav_io.py` — WAV files

- `save_wav(audio, path, sample_rate)` writes audio to a WAV file. It creates the folder if it doesn't exist.
- `load_wav(path)` reads a WAV file back into a NumPy array. It raises a `WavFileError` if the file is missing, broken, or not 16-bit.
- `wav_info(path)` returns the file's details without loading all the audio.

### `audio/levels.py` — measuring sound

- `rms_dbfs(audio)` gives the average loudness in dBFS.
- `audio_stats(audio, sample_rate)` gives duration, peak level, average level, and the number of clipped samples.
- `level_bar(db)` turns a dB number into the text bar you see in the live meter.

### `tests/test_audio.py` — tests

These tests check the parts that **don't need a microphone**, so they run anywhere, even on GitHub's servers. They check that:

- saving a WAV file and loading it back gives the exact same audio
- file details (channels, bit depth, length) are correct
- the wrong data type is rejected
- missing or broken files raise the right error
- a full-volume sine wave measures about -3 dBFS (this is a known fact about sine waves, so it's a good check that my math is right)
- clipping is counted correctly

---

## Setup

You'll need **Python 3.10 or newer**.

### 1. Get the code

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

A virtual environment is a private folder for this project's packages. It keeps them separate from other Python projects on your computer.

```bash
python -m venv .venv
```

Turn it on:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your terminal line when it's on.

### 3. Install the packages

```bash
pip install -r requirements.txt
```

This installs:
- **sounddevice** — talks to the mic and speakers
- **numpy** — stores and processes audio as arrays of numbers
- **pytest** — runs the tests

### 4. Linux only: install PortAudio

```bash
sudo apt install libportaudio2
```

On Windows and macOS, PortAudio usually comes bundled with `sounddevice`. If not on macOS, run `brew install portaudio`.

---

## How to Use It

Run the tester from the project folder:

```bash
python mic_tester.py
```

You'll see this menu:

```
=== Mic Recorder & Playback Tester ===
 1. List audio devices
 2. Play test tone (check speakers)
 3. Live mic level meter (check mic)
 4. Record and save WAV
 5. Play last recording
 6. Show last recording info
 0. Quit
```

### Suggested order the first time

1. **Option 1** — See your devices. Note the number of the mic and speaker you want.
2. **Option 2** — You should hear a beep. If yes, your speakers work.
3. **Option 3** — Talk. The bar should move. If it stays empty, the wrong mic is selected, or it's muted.
4. **Option 4** — Record yourself. Press Enter to use the default 5 seconds, or type a number. After recording, you'll see the levels and a quick quality check.
5. **Option 5** — Listen to your recording.
6. **Option 6** — See the file details.

Recordings are saved in the `temp/` folder with the date and time in the name, for example `recording_20260928_143015.wav`.

---

## Running the Tests

```bash
python -m pytest
```

You should see something like:

```
7 passed
```

Use `python -m pytest` instead of just `pytest`, so Python can find the `audio` package from the project folder.

---

## Settings You Can Change

Open `config.py`:

| Setting | Default | What it does |
|---------|---------|--------------|
| `SAMPLE_RATE` | `16000` | Samples per second |
| `CHANNELS` | `1` | 1 = mono, 2 = stereo |
| `DTYPE` | `"int16"` | Sample format (16-bit) |
| `RECORD_SECONDS` | `5` | Default recording length |
| `INPUT_DEVICE` | `None` | Which mic. `None` = system default. You can use a number from option 1, like `3`, or part of the name, like `"USB"` |
| `OUTPUT_DEVICE` | `None` | Which speaker, same rules as above |
| `TOO_QUIET_DBFS` | `-40.0` | Below this level, you get a "too quiet" warning |

---

## Problems I Hit and How to Fix Them

**"PortAudio library not found"**
The system audio library is missing. See step 4 of Setup.

**The level bar doesn't move**
- The wrong mic is probably selected. Run option 1, find your mic's number, and set `INPUT_DEVICE` in `config.py`.
- Check the mic isn't muted in your system settings.
- On macOS and Windows, check that your terminal is allowed to use the microphone (in privacy settings).

**"Invalid sample rate" or similar error**
Some mics don't support 16,000 Hz directly. Try another device from the list, or set `SAMPLE_RATE = 48000` to test. (For Week 2, I'll add a step to convert to 16 kHz if needed.)

**The recording sounds crackly**
That's probably clipping. The tester will show a warning. Move back from the mic or lower the mic volume in your system settings.

**The recording is very quiet**
Speak closer, or raise the mic volume in your system settings.

---

## Small Experiments to Try

These helped me understand the concepts better:

- Change `SAMPLE_RATE` to `44100`, record 5 seconds, and compare the file size with option 6. It should be about 2.75 times bigger.
- Change `CHANNELS` to `2` and see what happens to the file size.
- Whisper into the mic, then speak normally, then speak loudly. Watch how the dBFS number changes.
- In `player.py`, change the beep frequency from `440` to `880`. It will sound one octave higher.
- Break something in `wav_io.py` on purpose, then run the tests. See which ones fail.

---

## What's Next (Week 2)

Next week I'll add **speech recognition**. I'll use the audio recorded by this project and turn it into text using **Whisper**, running locally. The mini project will be a **voice-to-text note taker**: speak, and your words get saved as text notes.

Because I already record in the exact format Whisper wants (16 kHz, mono, 16-bit), the `audio/` package from this week can plug straight in.
