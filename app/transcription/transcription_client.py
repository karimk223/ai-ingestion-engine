"""Transcribe audio locally and return (start_seconds, text)."""
from functools import lru_cache
import subprocess

import imageio_ffmpeg
import numpy as np
import whisper
import re

from app.config import settings


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Keep the loaded model available for later transcription calls."""
    return whisper.load_model(model_name, device="cpu")


def transcribe_audio(
    audio_path: str, *, mixed_language: bool = False
) -> list[tuple[int, str]]:
    sample_rate = 16000
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-hide_banner", "-nostats",
        "-v", "info" if mixed_language else "error",
        "-i", str(audio_path),
        "-map", "0:a:0", "-vn",
    ]
    if mixed_language:
        command += ["-af", "silencedetect=noise=-35dB:d=1"]

    command += ["-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"]
    decoded = subprocess.run(
        command, capture_output=True, check=True, timeout=120
    )
    audio = np.frombuffer(decoded.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return []

    cuts = [0]
    if mixed_language:
        log = decoded.stderr.decode(errors="replace")
        pauses = re.findall(
            r"silence_start:\s*([\d.]+).*?silence_end:\s*([\d.]+)",
            log,
            flags=re.DOTALL,
        )
        duration = len(audio) / sample_rate
        for start, end in pauses:
            start, end = float(start), float(end)
            cut = round((start + end) / 2 * sample_rate)
            # Skip leading/trailing silence and avoid very short pieces.
            if start > 0 and end < duration and min(
                cut - cuts[-1], len(audio) - cut
            ) >= 2 * sample_rate:
                cuts.append(cut)
    cuts.append(len(audio))

    model = _get_model("small" if mixed_language else settings.WHISPER_MODEL)
    output = []
    for start, end in zip(cuts, cuts[1:]):
        result = model.transcribe(
            audio[start:end], task="transcribe", fp16=False, verbose=False
        )
        offset = start / sample_rate
        segments = [
            (int(offset + segment.get("start", 0)), segment["text"].strip())
            for segment in result.get("segments", [])
            if segment.get("text", "").strip()
        ]
        if segments:
            output.extend(segments)
        elif result.get("text", "").strip():
            output.append((int(offset), result["text"].strip()))

    return output