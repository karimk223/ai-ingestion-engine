import sys
from pathlib import Path

from app.services.ingestion_service import (
    _chunk,
    _file_id_from_path,
    _ingest_texts,
)
from app.vector_store.qdrant_store import ensure_collection

def ingest_document(document_path: str, video_path: str) -> dict:
    text = Path(document_path).read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("The reviewed document is empty.")

    video = Path(video_path)

    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    file_id = _file_id_from_path(str(video))
    chunks = _chunk(text)

    if not chunks:
        raise ValueError("The reviewed document produced no chunks.")

    ensure_collection()

    stored = _ingest_texts(
        file_id=file_id,
        file_name=video.name,
        source_type="video",
        chunks=chunks,
        page_numbers=[None] * len(chunks),
    )

    print(f"Stored {stored} chunks for {video.name}")
    print(f"File ID: {file_id}")

    return {
        "file_id": file_id,
        "file_name": video.name,
        "chunks_created": stored,
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python ingest_document.py DOCUMENT_PATH VIDEO_PATH"
        )

    ingest_document(sys.argv[1], sys.argv[2])