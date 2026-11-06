"""Transcribe audio locally and return (start_seconds, text)."""
from functools import lru_cache
import subprocess

import imageio_ffmpeg
import numpy as np
import whisper

from app.config import settings


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Keep the loaded model available for later transcription calls."""
    return whisper.load_model(model_name, device="cpu")


def transcribe_audio(audio_path: str) -> list[tuple[int, str]]:
    # Prepare the MP3 in the audio format Whisper reads.
    decoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin", "-v", "error",
            "-i", str(audio_path),
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )

    audio = np.frombuffer(decoded.stdout, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0

    if audio.size == 0:
        return []

    # Recognize the speech using the local model.
    model = _get_model(settings.WHISPER_MODEL)

    result = model.transcribe(
        audio,
        task="transcribe",
        fp16=False,
        verbose=False,
    )

    # Keep the same output format our pipeline already expects.
    output: list[tuple[int, str]] = []

    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()

        if text:
            output.append((int(segment.get("start", 0)), text))

    # If there is text but no separate sections, return it at second zero.
    if not output:
        text = result.get("text", "").strip()

        if text:
            output.append((0, text))

    return output
