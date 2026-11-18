import sys
from pathlib import Path

from app.config import settings
from app.parsers.video_parser import extract_video_frames
from app.parsers.audio_extractor import extract_audio
from app.transcription.transcription_client import transcribe_audio
from app.synthesis.document_assembler import assemble_document

def synthesize_video(video_path: str, output_path: str) -> None:
    print(f"Reading video frames: {video_path}")
    frames = extract_video_frames(video_path)

    transcript = []

    if settings.TRANSCRIBE_VIDEO:
        audio_path = extract_audio(video_path)

        if audio_path is not None:
            try:
                transcript = transcribe_audio(audio_path)
            finally:
                Path(audio_path).unlink(missing_ok=True)

    if not frames and not transcript:
        raise ValueError("No usable speech or on-screen content found.")

    if not transcript:
        print("No speech transcript available. Using on-screen content only.")
    elif not frames:
        print("No usable on-screen content available. Using speech only.")

    assembly = assemble_document(
        file_name=Path(video_path).name,
        frames=frames,
        transcript=transcript,
        max_chars=settings.SYNTHESIS_WINDOW_CHARS,
    )

    document = assembly["document"]

    Path(output_path).write_text(document, encoding="utf-8")
    print(f"Document saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python synthesize_video.py VIDEO_PATH [OUTPUT_PATH]"
        )

    video_path = sys.argv[1]

    if not Path(video_path).is_file():
        raise SystemExit(f"Video not found: {video_path}")
    
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f"{Path(video_path).stem}_document.md"
    )
    
    synthesize_video(video_path, output_path)