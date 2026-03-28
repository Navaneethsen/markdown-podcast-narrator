#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "InquirerPy>=0.3.4",
#     "markdown-it-py>=3.0.0",
#     "mdit-py-plugins>=0.4.0",
#     "click>=8.1.0",
#     "soundfile>=0.12.0",
#     "kokoro",
#     "misaki[en]",
# ]
# ///
"""Interactive CLI for Markdown Podcast Narrator.

Provides a file-picker prompt so you can browse and select
a Markdown file, then narrates it to MP3.
"""

import os
import sys
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from InquirerPy import inquirer

from parser import MarkdownParser
from narrator import Narrator

# Remember where the user ran the command from
INVOKE_DIR = Path(os.getcwd()).resolve()

PARENT_DIR = "📁 .."


def _list_entries(directory: Path) -> list[str]:
    """List subdirectories and .md files in a directory, sorted."""
    entries = [PARENT_DIR]
    dirs = []
    files = []
    try:
        for item in sorted(directory.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                dirs.append(f"📁 {item.name}/")
            elif item.suffix.lower() == ".md":
                files.append(f"📄 {item.name}")
    except PermissionError:
        pass
    return entries + dirs + files


def pick_markdown_file() -> Path:
    """Interactive directory browser. Arrow keys + Enter to navigate."""
    current = Path.home()

    while True:
        entries = _list_entries(current)
        choice = inquirer.select(
            message=f"  {current}",
            choices=entries,
            default=entries[1] if len(entries) > 1 else entries[0],
        ).execute()

        if choice == PARENT_DIR:
            current = current.parent
        elif choice.startswith("📁 "):
            dirname = choice[2:].rstrip("/").strip()
            current = current / dirname
        else:
            filename = choice[2:].strip()
            return (current / filename).resolve()


def main():
    selected_path = pick_markdown_file()
    output_file = INVOKE_DIR / selected_path.with_suffix(".wav").name

    print(f"\nInput:  {selected_path}")
    print(f"Output: {output_file}\n")

    # Parse
    print("Parsing markdown...")
    content = selected_path.read_text(encoding="utf-8")
    parser = MarkdownParser()
    tokens = parser.parse_to_speech_tokens(content)
    if not tokens:
        print("Error: no content found in markdown file", file=sys.stderr)
        sys.exit(1)

    # Init TTS (default: kokoro)
    engine = "kokoro"
    print(f"Initializing TTS ({engine})...")
    narrator = Narrator(engine=engine)

    if not narrator.initialize():
        print(f"{engine} unavailable, falling back to macOS 'say'...")
        narrator = Narrator(engine="macos")
        if not narrator.initialize():
            print("Error: no TTS backend available", file=sys.stderr)
            sys.exit(1)

    narrator.set_voice_params(rate=0.95)

    # Choose chunk strategy
    if narrator.is_neural:
        sections = parser.tokens_to_section_chunks(tokens)
        if not sections:
            print("Error: no speakable content", file=sys.stderr)
            sys.exit(1)

        total = sum(len(t) for t, _ in sections)
        print(f"Prepared {len(sections)} sections ({total} chars)")
        print("Generating audio (section-by-section)...")

        def on_progress(current, total):
            print(f"  Section {current}/{total}...", end="\r")

        ok = narrator.synthesize_sections(sections, str(output_file), on_progress)
        print()
    else:
        chunks = parser.tokens_to_speech_chunks(tokens)
        if not chunks:
            print("Error: no speakable content", file=sys.stderr)
            sys.exit(1)

        total = sum(len(t) for t, _ in chunks)
        print(f"Prepared {len(chunks)} chunks ({total} chars)")
        print("Generating audio...")
        ok = narrator.synthesize_chunks(chunks, str(output_file))

    if not ok:
        print("Error: audio generation failed", file=sys.stderr)
        sys.exit(1)

    size_kb = output_file.stat().st_size / 1024
    print(f"Saved to {output_file} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
