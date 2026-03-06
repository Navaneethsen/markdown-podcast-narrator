# Markdown Podcast Narrator

Converts a Markdown document into a high-quality podcast-style audio file using local TTS. Supports multiple backends — run fully offline on Apple Silicon with **Qwen3-TTS** or **Kokoro**, or use the built-in macOS voice as a zero-download fallback.

Please read the [medium article](https://navaneethsen.medium.com/i-built-a-tool-that-turns-any-document-into-a-podcast-in-a-weekend-using-ai-agents-3ae4867b3e2a) for the introduction

## What It Does

- Reads a `.md` file
- Parses headings, paragraphs, lists, code blocks, tables, and admonitions
- Converts each element into natural speech with expressive intros and transitions
- ALL-CAPS acronyms (API, HTTP, MAX_RETRIES) are automatically spelled out letter by letter
- Synthesizes section by section with consistent narrator tone
- Stitches all sections with calibrated silence gaps
- Outputs a single `.wav` file

## Audio Examples

| Engine | Sample |
|---|---|
| Qwen3-TTS | [▶ podcast-qwen.wav](<audio controls><source src="examples/podcast-qwen.wav" type="audio/mpeg"></audio>) |
| Kokoro | [▶ podcast-kokoro.wav](examples/podcast-kokoro.wav) |
| macOS say | [▶ podcast-macos.wav](examples/podcast-macos.wav) |




## Supported TTS Engines

| Engine | Flag | Quality | Speed | Requires |
|---|---|---|---|---|
| **Qwen3-TTS** | `--engine qwen` (default) | Excellent | Slow (~1-2s/section on MPS) | `qwen-tts torch` + `brew install sox` |
| **Kokoro** | `--engine kokoro` | Very good | Fast (CPU real-time) | `kokoro misaki[en]` |
| **macOS say** | `--engine macos` or `--fallback` | Good | Very fast | None (built-in) |

## Supported Markdown Elements

| Element | How It's Narrated |
|---|------|
| `# H1` | Chapter title with 1.2s pause after |
| `## H2` | Rotating transitions: "Now let's look at...", "Moving on to...", etc. |
| `### H3+` | Section title with 0.5s pause |
| Paragraphs | Read naturally with 0.4s gap |
| `- bullet list` | Detects if preceding paragraph already introduces it; "First, ..., Second, ..." |
| `1. ordered list` | Same as bullets with ordinal numbers |
| ` ```code``` ` | Announces language, reads lines with light symbol cleanup |
| `\| table \|` | Announces columns + row count; reads rows if 5 or fewer |
| `!!! info/warning/tip` | Expressive intro per type ("Here's a helpful tip...", "Critical alert...") |
| `---` | 1s pause between major sections |
| ALL_CAPS words | Spelled out: `API` → "A P I", `MAX_RETRIES` → "M A X underscore R E T R I E S" |

## Requirements

- **Python 3.10+**
- **uv** (recommended) or pip
- **macOS** for the `say` fallback; Qwen3-TTS and Kokoro work on any platform

## Quick Start

### 1. Install core dependencies

```bash
cd markdown_narrator
uv pip install -r requirements.txt
```

### 2. Pick your engine

**Option A — Qwen3-TTS** (best quality, Apple Silicon)
```bash
brew install sox
uv pip install qwen-tts torch torchaudio
uv run main.py your_doc.md
```

**Option B — Kokoro** (fast, CPU-based, good quality)
```bash
uv pip install kokoro misaki[en]
uv run main.py your_doc.md --engine kokoro
```

**Option C — macOS say** (instant, no download)
```bash
uv run main.py your_doc.md --fallback
```

## Usage

```
uv run main.py INPUT_FILE [OPTIONS]

Arguments:
  INPUT_FILE    Path to the Markdown (.md) file to convert

Options:
  -o, --output PATH        Output audio file path (default: output.wav)
  --engine [qwen|kokoro|macos]
                           TTS engine to use (default: qwen)
  --speaker TEXT           Qwen3-TTS speaker name (default: Ryan)
  --rate FLOAT             Speech rate multiplier 0.5-2.0 (default: 0.95)
  --fallback               Use macOS 'say' (same as --engine macos)
  --model TEXT             Override Qwen3-TTS HuggingFace model ID
  --instruct TEXT          Narrator style instruction for Qwen3-TTS
  --kokoro-voice TEXT      Kokoro voice name (default: af_heart)
  --help                   Show this message and exit
```

### Examples

```bash
# Default (Qwen3-TTS)
uv run main.py docs/guide.md

# Custom output path
uv run main.py docs/guide.md -o podcasts/guide.wav

# Use Kokoro (fast, CPU)
uv run main.py docs/guide.md --engine kokoro

# Slightly faster narration (~15% shorter audio)
uv run main.py docs/guide.md --rate 1.15

# Test without downloading any model
uv run main.py example.md --fallback

# Use the larger 1.7B Qwen model for higher quality
uv run main.py docs/guide.md --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Custom narrator tone
uv run main.py docs/guide.md --instruct "Speak slowly and warmly, like a bedtime story"

# Different Kokoro voice
uv run main.py docs/guide.md --engine kokoro --kokoro-voice af_sky
```

## Architecture

```
INPUT (.md)
    |
    v
MarkdownParser.parse_to_speech_tokens()
    — uses markdown-it-py + mdit-py-plugins
    — produces typed token list
    |
    v
MarkdownParser.tokens_to_speech_chunks() / tokens_to_section_chunks()
    — converts tokens to [(text, pause_ms)] chunks
    — speech_chunks: fine-grained (per paragraph/item) for macOS say
    — section_chunks: heading-grouped (per H1/H2) for neural TTS
    — pause_ms = silence to insert AFTER this chunk/section
    |
    v
Narrator.synthesize_sections() / synthesize_chunks()
    — Qwen3-TTS: section-level chunks with instruct for consistent tone
      calls model.generate_custom_voice(text, speaker, instruct)
    — Kokoro: section-level chunks via KPipeline
    — macOS fallback: single say call with [[slnc N]] embedded pauses
    — converts output to raw int16 PCM
    — stitches sections with silence frames between them
    — writes PCM incrementally to temp file (memory efficient)
    |
    v
_write_wav()  [stdlib struct only, no pydub]
    — writes RIFF/WAVE header + all PCM
    |
    v
OUTPUT (.wav)
```

### File responsibilities

| File | Purpose |
|---|---|
| `main.py` | CLI entry point. Orchestrates parse → chunk → synthesize pipeline. |
| `parser.py` | All markdown parsing and speech text generation. No audio logic here. |
| `narrator.py` | All audio generation. No markdown logic here. Handles all backends. |
| `pyproject.toml` | uv/pip project config. Core deps only; neural TTS engines are optional. |
| `requirements.txt` | Flat dependency list for `pip install -r`. |
| `example.md` | Test document covering all supported elements. |
| `AGENTS.md` | Instructions and context for AI agents extending this project. |

### Chunk pause durations

| Boundary | Silence |
|---|---|
| After H1 | 1200 ms |
| After H2 | 800 ms |
| After H3+ | 500 ms |
| After paragraph | 400 ms |
| Between list items | 200 ms |
| After list ends | 300 ms |
| After code/table block | 400 ms |
| After horizontal rule | 1000 ms |

## Available Qwen3-TTS Models

| Model ID | Size | Best for |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | **Default.** Fast, good quality, named speakers |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | Voice cloning from reference audio |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | Higher quality, slower |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | Describe voice in natural language |

## Available Kokoro Voices

Popular voices for American English (`lang_code="a"`):

| Voice | Character |
|---|---|
| `af_heart` | Default — warm female |
| `af_sky` | Light female |
| `am_adam` | Male |
| `am_michael` | Male, deeper |

Full list: [hexgrad/Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)

## Tuning for Large Documents

The `MAX_SECTION_CHARS` constant in `parser.py` (default: 600) controls how text is grouped into TTS calls:

- **Lower (300-500)** — more calls, more consistent emotion, safer memory usage
- **Higher (800-1200)** — fewer calls, faster overall, slight risk of prosody drift on very long sections

For a 55KB document with default settings, expect ~35-40 sections.

## Adding a New TTS Engine

1. Add `_init_<engine>(self) -> bool` to `Narrator` in `narrator.py`
2. Add `_synth_<engine>(self, text: str) -> Tuple[Optional[bytes], Optional[int]]` returning raw int16 PCM + sample rate
3. Add the engine name to `Narrator.ENGINES` and the `initialize()` dispatch
4. Add `_synth_one()` dispatch case
5. Add `--engine` choice to main.py's `click.option`

The chunking loop, silence insertion, memory management, and WAV writing are all engine-agnostic.

## Other Platforms

### Linux

The only macOS-specific parts are the `say` command fallback and `afconvert`. Qwen3-TTS and Kokoro both work natively on Linux with CUDA or CPU.

```bash
# Install dependencies
apt install sox
pip install qwen-tts torch torchaudio  # or: pip install kokoro misaki[en]
python main.py your_doc.md
```

For a CPU-only fallback TTS on Linux, add an `espeak-ng` backend:

```python
# narrator.py — Linux fallback example
def _init_espeak(self) -> bool:
    result = subprocess.run(["which", "espeak-ng"], capture_output=True)
    return result.returncode == 0

def _synth_espeak(self, text: str) -> Tuple[Optional[bytes], Optional[int]]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    subprocess.run(["espeak-ng", "-w", str(tmp_wav), "-s", "160", text], check=True)
    # read WAV PCM and return (pcm_bytes, sample_rate)
    ...
```

### Windows

- Replace `say`/`afconvert` with `pyttsx3` or Windows SAPI
- All path handling already uses `pathlib.Path`
- Qwen3-TTS and Kokoro work on Windows with CUDA or CPU

### Google Colab / Cloud GPU

```python
# Installs
!apt-get install -y sox
!pip install qwen-tts torch torchaudio

# Run
!python main.py your_doc.md
```

Qwen3-TTS auto-selects CUDA if available.

## Troubleshooting

**`ModuleNotFoundError: No module named 'qwen_tts'`**
```bash
uv pip install qwen-tts
```

**`ModuleNotFoundError: No module named 'kokoro'`**
```bash
uv pip install kokoro misaki[en]
```

**`sox: command not found` during Qwen3-TTS model load**
```bash
brew install sox
```

**Process killed / semaphore warning with large documents**
Lower `MAX_SECTION_CHARS` in `parser.py` (try 400-600). The process is running out of MPS GPU memory on large sections.

**Kokoro `words count mismatch` warnings**
Harmless — Kokoro's phonemizer logs these for acronyms and numbers. They are suppressed automatically.

**`say` fallback produces no output**
Test manually: `say -o /tmp/test.aiff "hello"`. If that fails, go to System Settings → Accessibility → Spoken Content and download a voice.

**Large document is slow with Qwen3-TTS**
- Switch to Kokoro: `--engine kokoro` (runs on CPU, much faster for batch work)
- On Apple Silicon, confirm MPS is active (look for `Loading ... on mps` in output)
- Use `--rate 1.1` for shorter audio output (~10% faster synthesis)

## Dependencies

| Package | Purpose |
|---|---|
| `markdown-it-py` | Markdown parsing (CommonMark spec) |
| `mdit-py-plugins` | Admonition (`!!!info`) support |
| `click` | CLI argument parsing |
| `soundfile` | Used by Qwen3-TTS internally |
| `qwen-tts` | Qwen3-TTS model loading and inference |
| `torch` | PyTorch for Qwen3-TTS inference |
| `torchaudio` | Audio tensor utilities |
| `kokoro` | Kokoro TTS engine |
| `misaki[en]` | English phonemizer for Kokoro |

No `pydub`, `ffmpeg`, or `librosa` required. WAV assembly uses Python stdlib `struct` only.

## License

MIT — see [LICENSE](LICENSE).
