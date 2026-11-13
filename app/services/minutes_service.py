"""
Minutes-of-Meeting service.

Given an ingested video's file_id, pull its content back out of Qdrant and
produce structured meeting minutes (Overview, Attendees, Key Points, Decisions,
Action Items) using the LLM.

Two ingestion modes are supported (mirrors extraction_service):
  - RAW mode: the file has separate transcript chunks (timestamp set, no frame
    number). Minutes are generated from the timestamped spoken content — a
    chronological view of the meeting.
  - SYNTHESIS mode: the file is one synthesized document, stored as narrative
    chunks with NO timestamp and NO frame number. Minutes are generated from
    the document text. This gives a summary-of-a-summary rather than a
    speech-by-speech reconstruction — the document has already stripped
    repetition, so minutes lose per-moment timing but gain coherence.

The context-window problem:
  A long meeting's content can exceed what the LLM can read in one call.
  So this uses MAP-REDUCE:
    - If the content fits in one call, summarize it directly (fast path).
    - If it's too long, split it into batches, summarize EACH batch ("map"),
      then combine those partial summaries into the final minutes ("reduce").
  This way the feature works for a 2-minute stand-up or a 2-hour meeting.

Token budgeting is approximate: we estimate ~4 characters per token and keep
each LLM call's input under a conservative character budget.
"""
import logging

from app.config import settings
from app.llm.groq_client import generate_answer
from app.vector_store.qdrant_store import get_chunks_by_file_id

logger = logging.getLogger(__name__)

# Conservative input budget per LLM call, in characters (~4 chars/token).
# Kept well under the model's real context window to leave room for the prompt
# and the generated output. Tune via .env if needed.
_CHARS_PER_BATCH = settings.MOM_BATCH_CHARS


_MINUTES_SYSTEM_PROMPT = (
    "You are an assistant that writes clear, professional Minutes of Meeting from "
    "meeting content. Produce well-structured minutes with these sections:\n"
    "1. Overview — a 1-2 sentence summary of the meeting.\n"
    "2. Attendees — names mentioned, if any (otherwise 'Not specified').\n"
    "3. Key Points — the main topics discussed, as bullet points.\n"
    "4. Decisions — any decisions made.\n"
    "5. Action Items — tasks assigned, with who is responsible if stated.\n"
    "Base the minutes ONLY on the content provided. Do not invent details. "
    "If a section has no content, write 'None noted.'"
    "\n\nTutorial examples and demo assignments belong in Key Points. "
    "Include only explicit real meeting decisions and commitments in Decisions "
    "and Action Items. If there are none, write 'None noted.'"
)

_REDUCE_SYSTEM_PROMPT = (
    "You are combining several partial meeting-minutes summaries (each covering a "
    "part of the same meeting, in order) into ONE final set of Minutes of Meeting. "
    "Merge them into a single coherent document with these sections: Overview, "
    "Attendees, Key Points, Decisions, Action Items. Remove duplication, keep it "
    "consistent and professional. Do not invent details beyond the partial summaries."
    "\n\nTutorial examples and demo assignments belong in Key Points. "
    "Include only explicit real meeting decisions and commitments in Decisions "
    "and Action Items. If there are none, write 'None noted.'"
)


def _batch_content(chunks: list[dict]) -> list[str]:
    """
    Group ordered content chunks into batches that each fit the char budget.

    Returns a list of content-text batches (each a big string).
    """
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for c in chunks:
        text = c.get("text", "").strip()
        if not text:
            continue
        if current_len + len(text) > _CHARS_PER_BATCH and current:
            batches.append("\n".join(current))
            current = []
            current_len = 0
        current.append(text)
        current_len += len(text)
    if current:
        batches.append("\n".join(current))
    return batches


def generate_minutes(file_id: str) -> dict:
    """Generate meeting minutes from a video's stored text."""
    chunks = get_chunks_by_file_id(file_id, source_type="video")

    transcript = [
        chunk for chunk in chunks
        if chunk.get("timestamp_seconds") is not None
        and chunk.get("frame_number") is None
    ]

    if transcript:
        source_chunks = sorted(
            transcript, key=lambda chunk: chunk.get("timestamp_seconds") or 0
        )
        source = "raw transcript"
    else:
        source_chunks = [
            chunk for chunk in chunks
            if chunk.get("frame_number") is None and (chunk.get("text") or "").strip()
        ]
        source = "synthesis document"

    if not source_chunks:
        raise RuntimeError(
            f"No usable content found for file_id={file_id}. "
            "Was the video ingested with transcription or synthesis enabled?"
        )

    file_name = source_chunks[0].get("file_name") or "unknown"
    batches = _batch_content(source_chunks)
    batch_count = len(batches)

    if batch_count == 1:
        method = "single-pass"
        minutes = generate_answer(
            system_prompt=_MINUTES_SYSTEM_PROMPT,
            user_prompt=(
                f"Meeting content ({source}):\n\n{batches[0]}\n\n"
                "Write the Minutes of Meeting."
            ),
        )
    else:
        method = "map-reduce"
        partials = []
        for index, batch in enumerate(batches, start=1):
            partial = generate_answer(
                system_prompt=_MINUTES_SYSTEM_PROMPT,
                user_prompt=(
                    f"This is part {index} of {batch_count} of the meeting content ({source}).\n\n"
                    f"{batch}\n\nWrite partial Minutes of Meeting for THIS part only."
                ),
            )
            partials.append(f"--- Partial minutes {index} ---\n{partial.strip()}")

        combined = "\n\n".join(partials)
        minutes = generate_answer(
            system_prompt=_REDUCE_SYSTEM_PROMPT,
            user_prompt=(
                f"Combine these {len(partials)} partial meeting-minutes summaries into "
                f"one final Minutes of Meeting:\n\n{combined}"
            ),
        )

    return {
        "file_id": file_id,
        "file_name": file_name,
        "minutes": minutes.strip(),
        "batches_used": batch_count,
        "method": method,
        "source": source,
    }