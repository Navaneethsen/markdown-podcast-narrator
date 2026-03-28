"""
TTS Narrator Module using multiple backends (macOS say, Qwen3-TTS, Kokoro).

Synthesis strategies per backend:

  macOS 'say':
    Sends the ENTIRE document as a single call with [[slnc N]] embedded
    pause commands. This preserves consistent prosody and emotion
    across the whole narration.

  Qwen3-TTS (neural TTS):
    Uses section-level chunks (grouped by heading) rather than
    per-paragraph chunks. Each section is ~100–500 characters of
    flowing text — long enough for the model to establish consistent
    emotion, but short enough to avoid attention drift. Sections are
    stitched together with explicit PCM silence. An optional
    `instruct` parameter enforces a consistent narrator tone across
    sections.

  Kokoro (fast neural TTS):
    Designed for low-latency local inference. The document is processed
    in medium-sized chunks (~200–400 characters) to keep generation fast
    while maintaining natural phrasing. Audio segments are concatenated
    with short pauses between sections to preserve readability and flow.
    Kokoro is significantly faster than large neural models and works
    well for quick narration, though the voice may be slightly less
    expressive than Qwen3-TTS.
"""

import logging
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SpeechChunk = Tuple[str, int]

# Default narrator instruction for consistent professional tone
DEFAULT_INSTRUCT = "Calm, professional podcast narrator. Speak at a brisk, confident pace of about 160 words per minute."


class Narrator:
    """Text-to-speech narrator with backend-appropriate synthesis strategy."""

    ENGINES = ("qwen", "kokoro", "macos")

    def __init__(self, engine: str = "qwen", model_id: Optional[str] = None,
                 use_qwen: bool = True):
        # engine param takes priority; use_qwen kept for backward compat
        if engine != "qwen":
            self.engine = engine
        elif not use_qwen:
            self.engine = "macos"
        else:
            self.engine = "qwen"

        self.model_id = model_id or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        self.model = None        # Qwen3-TTS model
        self.kokoro_pipeline = None  # Kokoro pipeline
        self.kokoro_voice = "af_heart"
        self.kokoro_speed = 1.0
        self.speaker = "Ryan"
        self.language = "English"
        self.say_rate = 180
        self.instruct: Optional[str] = DEFAULT_INSTRUCT

    @property
    def use_qwen(self) -> bool:
        return self.engine == "qwen"

    @property
    def is_neural(self) -> bool:
        """True if using a neural TTS engine (section-level chunking)."""
        if self.engine == "qwen":
            return self.model is not None
        if self.engine == "kokoro":
            return self.kokoro_pipeline is not None
        return False

    def initialize(self) -> bool:
        init_fns = {
            "qwen": self._init_qwen,
            "kokoro": self._init_kokoro,
            "macos": self._init_macos_say,
        }
        fn = init_fns.get(self.engine)
        if not fn:
            logger.error(f"Unknown engine: {self.engine}")
            return False
        return fn()

    # The CLI default rate (0.95) maps to 160 WPM — our baseline.
    _DEFAULT_RATE = 0.95

    def set_voice_params(self, rate: float = 0.95, speaker: str = "Ryan",
                         instruct: Optional[str] = None,
                         kokoro_voice: Optional[str] = None):
        self.speaker = speaker
        self.say_rate = max(90, min(720, int(180 * rate)))
        if kokoro_voice is not None:
            self.kokoro_voice = kokoro_voice
        if self.engine == "kokoro":
            self.kokoro_speed = rate / self._DEFAULT_RATE  # normalize: 0.95 → 1.0

        if instruct is not None:
            # User-supplied instruct overrides everything
            self.instruct = instruct
        elif rate != self._DEFAULT_RATE:
            # User changed --rate from the default — scale WPM accordingly.
            # 0.95 → 160 WPM (baseline), 1.15 → ~194 WPM, 0.7 → ~118 WPM
            target_wpm = int(160 * rate / self._DEFAULT_RATE)
            self.instruct = (
                f"Calm, professional podcast narrator. "
                f"Speak at a confident pace of about {target_wpm} words per minute."
            )
        # else: keep DEFAULT_INSTRUCT (160 WPM) from __init__

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def synthesize_chunks(
        self,
        chunks: List[SpeechChunk],
        output_file: Union[str, Path],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Synthesize fine-grained chunks. Used for macOS 'say' backend
        which merges them into a single call with [[slnc]] markers.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Always generate WAV first, then convert if needed
        want_mp3 = output_file.suffix.lower() == ".mp3"
        wav_file = output_file.with_suffix(".wav") if want_mp3 else output_file

        if self.engine == "macos":
            ok = self._synth_single_macos(chunks, wav_file)
        else:
            ok = self._synth_chunked(chunks, wav_file, on_progress)

        if ok and want_mp3:
            ok = _wav_to_mp3(wav_file, output_file)
        return ok

    def synthesize_sections(
        self,
        sections: List[SpeechChunk],
        output_file: Union[str, Path],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Synthesize section-level chunks for neural TTS engines.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        want_mp3 = output_file.suffix.lower() == ".mp3"
        wav_file = output_file.with_suffix(".wav") if want_mp3 else output_file

        if self.engine == "macos":
            ok = self._synth_single_macos(sections, wav_file)
        else:
            ok = self._synth_chunked(sections, wav_file, on_progress)

        if ok and want_mp3:
            ok = _wav_to_mp3(wav_file, output_file)
        return ok

    # ------------------------------------------------------------------
    # macOS 'say': single call with [[slnc]] embedded pauses
    # ------------------------------------------------------------------

    def _synth_single_macos(
        self,
        chunks: List[SpeechChunk],
        output_file: Path,
    ) -> bool:
        parts = []
        for text, pause_ms in chunks:
            parts.append(_prepare_text_for_tts(text))
            if pause_ms > 0:
                parts.append(f" [[slnc {pause_ms}]] ")

        full_text = " ".join(parts)

        logger.info(f"Synthesizing as single narration ({len(full_text)} chars)...")

        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                tmp_aiff = Path(tmp.name)

            cmd = ["say", "-o", str(tmp_aiff), "-r", str(self.say_rate)]
            proc = subprocess.run(
                cmd, input=full_text, capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0:
                logger.error(f"'say' failed: {proc.stderr}")
                return False

            result = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", str(tmp_aiff), str(output_file)],
                capture_output=True, text=True, check=False,
            )
            tmp_aiff.unlink(missing_ok=True)

            if result.returncode != 0:
                logger.error(f"afconvert failed: {result.stderr}")
                return False

            logger.info(f"Audio saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"macOS synth error: {e}")
            return False

    # ------------------------------------------------------------------
    # Generic chunked synthesis (used by all neural engines)
    # ------------------------------------------------------------------

    def _synth_one(self, text: str) -> Tuple[Optional[bytes], Optional[int]]:
        """Dispatch a single text to the active engine's synth method."""
        text = _prepare_text_for_tts(text)
        if self.engine == "qwen":
            return self._synth_qwen(text)
        if self.engine == "kokoro":
            return self._synth_kokoro(text)
        return None, None

    def _synth_chunked(
        self,
        sections: List[SpeechChunk],
        output_file: Path,
        on_progress: Optional[Callable[[int, int], None]],
    ) -> bool:
        import gc

        total = len(sections)
        tmp_pcm = output_file.with_suffix(".pcm.tmp")
        sample_rate = None
        written_any = False

        try:
            with open(tmp_pcm, "wb") as pcm_out:
                for i, (text, pause_ms) in enumerate(sections):
                    if on_progress:
                        on_progress(i + 1, total)

                    logger.info(f"Section {i+1}/{total} ({len(text)} chars)")
                    pcm_data, sr = self._synth_one(text)

                    if pcm_data is None:
                        logger.error(f"Section {i+1}/{total} failed: {text[:60]}...")
                        continue

                    sample_rate = sr
                    pcm_out.write(pcm_data)
                    written_any = True
                    del pcm_data

                    if pause_ms > 0:
                        pcm_out.write(_make_silence(pause_ms, sr))

                    gc.collect()
                    self._clear_device_cache()

            if not written_any or sample_rate is None:
                logger.error("No audio segments produced")
                tmp_pcm.unlink(missing_ok=True)
                return False

            pcm_data = tmp_pcm.read_bytes()
            _write_wav(output_file, pcm_data, sample_rate)
            tmp_pcm.unlink(missing_ok=True)
            logger.info(f"Audio saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            tmp_pcm.unlink(missing_ok=True)
            return False

    def _clear_device_cache(self):
        """Free cached GPU/MPS memory to prevent accumulation across sections."""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _synth_qwen(self, text: str) -> Tuple[Optional[bytes], Optional[int]]:
        try:
            import numpy as np
            import torch

            with torch.inference_mode():
                # Try with instruct parameter for consistent tone
                try:
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=self.language,
                        speaker=self.speaker,
                        instruct=self.instruct,
                    )
                except TypeError:
                    # Fallback if this model variant doesn't support instruct
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=self.language,
                        speaker=self.speaker,
                    )

            audio = np.array(wavs[0], dtype=np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            pcm = (audio * 32767).astype(np.int16).tobytes()
            return pcm, sr

        except Exception as e:
            logger.error(f"Qwen3-TTS synth error: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Single-string convenience
    # ------------------------------------------------------------------

    def text_to_audio(self, text: str, output_file: Union[str, Path]) -> bool:
        return self.synthesize_chunks([(text, 0)], output_file)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_qwen(self) -> bool:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            # Suppress verbose per-call noise from qwen_tts and transformers
            import logging as _logging
            _logging.getLogger("qwen_tts").setLevel(_logging.ERROR)
            _logging.getLogger("transformers").setLevel(_logging.ERROR)

            if torch.backends.mps.is_available():
                device, dtype = "mps", torch.bfloat16
            elif torch.cuda.is_available():
                device, dtype = "cuda:0", torch.bfloat16
            else:
                device, dtype = "cpu", torch.float32

            logger.info(f"Loading {self.model_id} on {device}...")

            kwargs = dict(device_map=device, dtype=dtype)
            if device.startswith("cuda"):
                try:
                    import flash_attn  # noqa: F401
                    kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    pass

            self.model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
            logger.info("Qwen3-TTS loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install with: pip install -U qwen-tts")
            return False
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
            return False

    # ------------------------------------------------------------------
    # Kokoro TTS
    # ------------------------------------------------------------------

    def _init_kokoro(self) -> bool:
        try:
            from kokoro import KPipeline
            import logging as _logging
            _logging.getLogger("misaki").setLevel(_logging.ERROR)

            lang = self.language.lower()
            lang_code = "a" if lang in ("english", "en", "a") else "a"
            logger.info(f"Loading Kokoro (lang={lang_code}, voice={self.kokoro_voice})...")

            self.kokoro_pipeline = KPipeline(lang_code=lang_code)
            logger.info("Kokoro loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install with: pip install kokoro soundfile")
            return False
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            return False

    def _synth_kokoro(self, text: str) -> Tuple[Optional[bytes], Optional[int]]:
        try:
            import numpy as np

            sample_rate = 24000
            audio_parts = []

            generator = self.kokoro_pipeline(
                text,
                voice=self.kokoro_voice,
                speed=self.kokoro_speed,
            )
            for _gs, _ps, audio in generator:
                if audio is not None:
                    audio_parts.append(audio)

            if not audio_parts:
                logger.error("Kokoro produced no audio")
                return None, None

            audio = np.concatenate(audio_parts)
            audio = np.clip(audio, -1.0, 1.0)
            pcm = (audio * 32767).astype(np.int16).tobytes()
            return pcm, sample_rate

        except Exception as e:
            logger.error(f"Kokoro synth error: {e}")
            return None, None

    def _init_macos_say(self) -> bool:
        try:
            result = subprocess.run(["which", "say"], capture_output=True, check=False)
            ok = result.returncode == 0
            if ok:
                logger.info("macOS 'say' available")
            else:
                logger.error("macOS 'say' not found")
            return ok
        except Exception as e:
            logger.error(f"Error checking 'say': {e}")
            return False


# ------------------------------------------------------------------
# Text preprocessing for TTS
# ------------------------------------------------------------------

def _prepare_text_for_tts(text: str) -> str:
    """Make text more pronounceable for TTS engines.

    - ALL_CAPS words (2+ uppercase letters) are spelled out letter by letter.
    - Underscores within such words become "underscore".
    - Digits within such words are spelled individually.

    Examples:
        "the API is ready"       → "the A P I is ready"
        "set MAX_RETRIES to 5"   → "set M A X underscore R E T R I E S to 5"
        "uses HTTP2 protocol"    → "uses H T T P 2 protocol"
    """
    import re

    def _spell(match: re.Match) -> str:
        word = match.group(0)
        parts = word.split("_")
        spelled = []
        for part in parts:
            if part:
                spelled.append(" ".join(part))
        return " underscore ".join(spelled)

    # Match words that are entirely uppercase + digits + underscores,
    # starting with an uppercase letter, at least 2 characters total.
    return re.sub(r'\b[A-Z][A-Z0-9_]+\b', _spell, text)


# ------------------------------------------------------------------
# Audio utilities (stdlib only)
# ------------------------------------------------------------------

def _make_silence(duration_ms: int, sample_rate: int) -> bytes:
    n_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n_samples


def _has_lame() -> bool:
    try:
        subprocess.run(["lame", "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def _wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "128") -> bool:
    """Convert a WAV file to MP3 using lame, then remove the WAV."""
    if not _has_lame():
        print("\nMP3 output requires 'lame' which is not installed.")
        print("Install it with:  brew install lame")
        input("\nPress Enter after installing lame to continue...")

        if not _has_lame():
            logger.error("lame still not found — keeping WAV file: " + str(wav_path))
            return False

    result = subprocess.run(
        ["lame", "-b", bitrate, str(wav_path), str(mp3_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        wav_path.unlink(missing_ok=True)
        return True
    logger.error(f"lame failed: {result.stderr}")
    return False


def _write_wav(path: Path, pcm_data: bytes, sample_rate: int,
               channels: int = 1, sample_width: int = 2):
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_data)
