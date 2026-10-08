"""Test how well Whisper understands YOUR voice, and compare model sizes.

Step 1 - record test samples (you read sentences out loud):
    python evaluate_stt.py record

Step 2 - measure accuracy and speed:
    python evaluate_stt.py run
    python evaluate_stt.py run --models tiny.en base.en small.en

Each sample is a pair of files in tests/stt_samples/:
    sample_01.wav   (your voice)
    sample_01.txt   (what you actually said = the "reference")
"""

import argparse
import sys
from pathlib import Path

import config
from audio.exceptions import AudioError
from audio.recorder import record_until_silence
from audio.wav_io import save_wav
from stt.exceptions import STTError
from stt.metrics import word_error_rate
from stt.whisper_stt import WhisperSTT

# A mix of easy sentences, numbers, names and tech words. The hard ones show
# where Whisper struggles, which is exactly what we want to find out.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Add milk, eggs and bread to my shopping list.",
    "Set a timer for ten minutes.",
    "Remind me to call Priya at five thirty tomorrow.",
    "Whisper turns speech into text on my own computer.",
    "Ollama runs large language models locally.",
    "Open the file called config dot py.",
    "What is twenty seven times fourteen?",
]


def find_samples(folder: Path) -> list[tuple[Path, str]]:
    pairs = []
    for wav in sorted(folder.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt.read_text(encoding="utf-8").strip()))
        else:
            print(f"Skipping {wav.name}: no matching .txt file")
    return pairs


def next_index(folder: Path) -> int:
    numbers = [int(p.stem.split("_")[-1]) for p in folder.glob("sample_*.wav")
               if p.stem.split("_")[-1].isdigit()]
    return max(numbers, default=0) + 1


def cmd_record(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    print("Read each sentence out loud. Press Enter to start each one.")
    print("Type 's' to skip a sentence, or 'q' to quit.\n")

    for sentence in SENTENCES:
        print(f'Say: "{sentence}"')
        answer = input("  [Enter] record / s / q: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        try:
            audio, _ = record_until_silence(
                on_event=lambda e: print("  >> Speak now." if e == "calibrated" else "  >> Hearing you..."))
        except AudioError as exc:
            print(f"  Audio error: {exc}")
            return
        if audio is None:
            print("  Didn't hear anything, skipping.")
            continue
        idx = next_index(folder)
        wav = save_wav(audio, folder / f"sample_{idx:02d}.wav", config.SAMPLE_RATE)
        wav.with_suffix(".txt").write_text(sentence + "\n", encoding="utf-8")
        print(f"  Saved {wav.name}\n")


def cmd_run(folder: Path, models: list[str]) -> None:
    samples = find_samples(folder)
    if not samples:
        print(f"No samples in {folder}. Run: python evaluate_stt.py record")
        return

    summary = []
    for model_name in models:
        print(f"\n=== Model: {model_name} ===")
        stt = WhisperSTT(model_size=model_name)
        try:
            stt.load()
        except STTError as exc:
            print(f"Error: {exc}")
            continue
        print(f"Loaded in {stt.load_seconds:.1f}s\n")

        total_errors = total_words = 0
        total_audio = total_time = 0.0
        for wav, reference in samples:
            result = stt.transcribe_file(wav)
            score = word_error_rate(reference, result.text)
            total_errors += score.errors
            total_words += score.reference_words
            total_audio += result.audio_seconds
            total_time += result.processing_seconds
            print(f"{wav.name}  WER {score.wer:6.1%}  RTF {result.rtf:.2f}")
            if score.errors:
                print(f"    expected: {reference}")
                print(f"    got:      {result.text}")

        # Overall WER = all errors / all words (not an average of percentages,
        # so long sentences count more than short ones).
        wer = total_errors / total_words if total_words else 0.0
        rtf = total_time / total_audio if total_audio else 0.0
        summary.append((model_name, wer, rtf, stt.load_seconds))

    if summary:
        print("\n=== Summary ===")
        print(f"{'Model':<12} {'WER':>7} {'RTF':>6} {'Load':>7}")
        for name, wer, rtf, load in summary:
            print(f"{name:<12} {wer:>7.1%} {rtf:>6.2f} {load:>6.1f}s")
        print("\nWER: lower = more accurate. RTF: lower = faster (below 1.0 = faster than real time).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and evaluate Whisper test samples.")
    parser.add_argument("--folder", type=Path, default=config.STT_SAMPLES_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("record", help="record test sentences")
    run = sub.add_parser("run", help="measure accuracy and speed")
    run.add_argument("--models", nargs="+", default=[config.WHISPER_MODEL])
    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args.folder)
    else:
        cmd_run(args.folder, args.models)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        sys.exit(1)
