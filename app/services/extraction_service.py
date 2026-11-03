"""Extract connections from a stored PDF or video's text."""
import logging

from app.extraction.entity_extractor import extract_triples
from app.vector_store.qdrant_store import get_chunks_by_file_id


logger = logging.getLogger(__name__)


def extract_from_file(file_id: str) -> dict:
    """Read a file's stored chunks and extract its connections."""
    chunks = get_chunks_by_file_id(file_id)

    if not chunks:
        raise RuntimeError(f"No stored chunks found for file_id '{file_id}'.")

    transcript = [
        c for c in chunks
        if c.get("timestamp_seconds") is not None
        and c.get("frame_number") is None
    ]

    if transcript:
        transcript.sort(key=lambda c: c.get("timestamp_seconds") or 0)
        source_chunks = transcript
        mode = "raw transcript"
    else:
        source_chunks = [
            c for c in chunks
            if c.get("frame_number") is None and (c.get("text") or "").strip()
        ]
        mode = "document text"

    if not source_chunks:
        raise RuntimeError(f"No usable text found for file_id '{file_id}'.")

    file_name = source_chunks[0].get("file_name")
    segments = [c.get("text", "") for c in source_chunks]

    logger.info(
        "Extracting connections from '%s' (%d chunks, %s)",
        file_name,
        len(segments),
        mode,
    )
    result = extract_triples(segments)

    return {
        "file_id": file_id,
        "file_name": file_name,
        **result,
    }
