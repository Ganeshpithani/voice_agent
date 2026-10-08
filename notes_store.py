"""Save notes as simple Markdown files, one file per day: notes/2026-09-29.md"""

from datetime import date, datetime
from pathlib import Path

import config


def notes_path(day: date, notes_dir: Path = config.NOTES_DIR) -> Path:
    return Path(notes_dir) / f"{day.isoformat()}.md"


def save_note(text: str, when: datetime | None = None,
              notes_dir: Path = config.NOTES_DIR) -> Path:
    """Append one note to that day's file. Creates the file with a title if needed."""
    text = " ".join(text.split())  # one line, no extra spaces
    if not text:
        raise ValueError("Cannot save an empty note")
    when = when or datetime.now()
    path = notes_path(when.date(), notes_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# Notes — {when.date().isoformat()}\n\n")
        f.write(f"- **{when:%H:%M:%S}** — {text}\n")
    return path


def read_notes(day: date | None = None, notes_dir: Path = config.NOTES_DIR) -> str | None:
    """Return that day's notes as text, or None if there are none."""
    path = notes_path(day or date.today(), notes_dir)
    return path.read_text(encoding="utf-8") if path.exists() else None
