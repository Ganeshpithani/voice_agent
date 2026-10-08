"""Week 2 mini project: voice-to-text note taker.

Speak -> it records until you stop -> Whisper turns it into text -> saved as a note.

Run from the project folder:
    python note_taker.py
"""

from datetime import datetime
from pathlib import Path

import config
from audio.exceptions import AudioError
from audio.levels import audio_stats
from audio.recorder import record_until_silence
from audio.wav_io import save_wav
from notes_store import read_notes, save_note
from stt.exceptions import STTError
from stt.whisper_stt import TranscriptResult, WhisperSTT

MENU = """
=== Voice Note Taker ===
 1. Record a note (stops when you stop talking)
 2. Transcribe a WAV file
 3. Show today's notes
 4. Show settings
 0. Quit
"""


def on_event(name: str) -> None:
    if name == "calibrated":
        print(">> Listening... speak now.")
    elif name == "speech_start":
        print(">> Hearing you... (stop talking to finish)")


def print_result(result: TranscriptResult) -> None:
    print("\n--- Transcript ---")
    print(result.text or "(no words recognised)")
    print("------------------")
    lang = f"{result.language} ({result.language_probability:.0%})" if result.language else "?"
    print(f"Language: {lang} | Audio: {result.audio_seconds:.1f}s | "
          f"Took: {result.processing_seconds:.2f}s | RTF: {result.rtf:.2f}")
    if result.dropped_segments:
        print(f"(Ignored {result.dropped_segments} part(s) that looked like silence/noise)")


def confirm_and_save(text: str) -> None:
    if not text:
        return
    answer = input("Save this note? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        path = save_note(text)
        print(f"Saved to {path}")
    else:
        print("Not saved.")


def record_note(stt: WhisperSTT) -> None:
    print("Stay quiet for a moment while I measure the room noise...")
    audio, reason = record_until_silence(on_event=on_event)

    if audio is None:
        print("I didn't hear anything. Check your mic with mic_tester.py (option 3).")
        return
    if reason == "max_length":
        print(f"(Stopped at the {config.MAX_RECORD_SECONDS:.0f}s limit)")

    stats = audio_stats(audio, config.SAMPLE_RATE)
    if stats["clipped"]:
        print("Warning: some audio was too loud (clipping). Results may be worse.")

    if config.KEEP_NOTE_AUDIO:
        path = config.TEMP_DIR / f"note_{datetime.now():%Y%m%d_%H%M%S}.wav"
        save_wav(audio, path, config.SAMPLE_RATE)

    print("Transcribing...")
    result = stt.transcribe(audio, config.SAMPLE_RATE)
    print_result(result)
    confirm_and_save(result.text)


def transcribe_file(stt: WhisperSTT) -> None:
    path = Path(input("Path to WAV file: ").strip().strip('"'))
    print("Transcribing...")
    result = stt.transcribe_file(path)
    print_result(result)
    confirm_and_save(result.text)


def show_settings(stt: WhisperSTT) -> None:
    print(f"Model:        {stt.model_size} on {stt.device} ({stt.compute_type})")
    print(f"Language:     {stt.language or 'auto-detect'}")
    print(f"Beam size:    {stt.beam_size}")
    print(f"Load time:    {stt.load_seconds:.1f}s")
    print(f"Silence stop: {config.SILENCE_SECONDS}s | Max length: {config.MAX_RECORD_SECONDS:.0f}s")
    print(f"Notes folder: {config.NOTES_DIR}")


def main() -> None:
    stt = WhisperSTT()
    print(f"Loading Whisper model '{stt.model_size}' "
          "(the first run downloads it, this can take a minute)...")
    try:
        stt.load()
    except STTError as exc:
        print(f"Error: {exc}")
        return
    print(f"Model ready in {stt.load_seconds:.1f}s.")

    while True:
        print(MENU)
        choice = input("Choose: ").strip()
        try:
            if choice == "1":
                record_note(stt)
            elif choice == "2":
                transcribe_file(stt)
            elif choice == "3":
                print(read_notes() or "No notes today yet.")
            elif choice == "4":
                show_settings(stt)
            elif choice == "0":
                print("Bye!")
                break
            else:
                print("Unknown option.")
        except (AudioError, STTError) as exc:
            print(f"Error: {exc}")
        except ValueError as exc:
            print(f"Bad input: {exc}")
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
