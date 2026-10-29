"""
Synthesis agent — Step 2 of the video-synthesis pipeline.

Takes ONE window (the interleaved transcript + frame descriptions for a slice of
the video, produced by window_splitter) and writes a clean, grounded Markdown
narrative of that slice: what was said and what was on screen, woven together in
time order.

Design decisions (settled with instructor):
- ONE synthesis agent per window (frames are already described upstream by the
  vision model, so no separate frame-agent is needed).
- Output is flowing NARRATIVE PROSE, not a rigid template. Rigid templates create
  empty slots the model feels obliged to fill, which invites fabrication when a
  window is sparse. Prose lets the model be honest about how little is there.
- Timestamps are NOT required in the output (per instructor) — natural prose.

Overlap / continuity:
  A window can be cut mid-topic (the boundary is by size, not meaning), so the
  next window may begin in the middle of a thought. To soften that, a window may
  carry a small read-only RECAP of the previous window (window.context). The agent
  is told to use the recap ONLY to understand what follows — it must NOT re-describe
  the recap, or the seams would fill with duplicated sentences. This gives the
  agent continuity without causing repetition.

ANTI-HALLUCINATION is the core concern. The prompt is strict: describe ONLY what
is in the provided speech and frames, never infer, never invent, and say the
material is unclear rather than guess.

The LLM client is injected so this is unit-testable without real API calls.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.synthesis.window_splitter import Window

logger = logging.getLogger(__name__)


_SYNTHESIS_PROMPT = (
    "You are writing a faithful narrative of one short segment of a recorded "
    "video (a meeting or software walkthrough). You are given up to three things "
    "for this segment:\n"
    "  - RECAP (optional): a short reminder of what was said/shown at the very end "
    "of the PREVIOUS segment. This is context only — it helps you understand what "
    "the current segment continues from.\n"
    "  - SPEECH: what was said (transcript lines, in order; may be in Arabic, "
    "English, or mixed).\n"
    "  - ON SCREEN: descriptions of what was visible on screen (from a vision "
    "model reading each frame).\n\n"
    "Write 1-2 short paragraphs of plain narrative describing what happens in this "
    "segment — weaving together what was said and what was shown, in the order it "
    "occurred. Write it so a person could read it and understand this part of the "
    "video without watching it.\n\n"
    "CRITICAL RULES — follow exactly:\n"
    "- Describe ONLY the current segment's SPEECH and ON SCREEN content. The RECAP "
    "is background for your understanding — do NOT re-describe or repeat it; if the "
    "current segment simply continues the recap, carry on from where it left off.\n"
    "- Use ONLY the information provided. Do NOT add facts, explanations, or "
    "background from your own knowledge.\n"
    "- Do NOT infer intentions, conclusions, or connections that are not directly "
    "supported by the given text.\n"
    "- If the segment is sparse or unclear, write only the little that is certain "
    "and say it is brief or unclear — do NOT pad it out or invent detail.\n"
    "- Do not guess the meaning of a technical term from its name; if its meaning "
    "isn't shown, just report that the term appeared.\n"
    "- Keep any product names, field names, and numbers exactly as given.\n"
    "- Write in English. Do not use headings or bullet points — plain paragraphs.\n"
    "- Output ONLY the narrative. No preamble, no 'In this segment...' boilerplate."
)


def _format_window_for_prompt(window: Window) -> str:
    """Render a window's events into the RECAP / SPEECH / ON SCREEN blocks."""
    speech = [e.text for e in window.events if e.kind == "speech"]
    frames = [e.text for e in window.events if e.kind == "frame"]

    speech_block = "\n".join(f"- {s}" for s in speech) if speech else "(no speech in this segment)"
    frames_block = "\n".join(f"- {f}" for f in frames) if frames else "(nothing readable on screen)"

    blocks = ""
    # Only include a RECAP block when there is carried-over context (i.e. not the
    # first window). The recap mixes the previous window's speech + on-screen tail.
    if window.context:
        recap_lines = "\n".join(f"- {e.text}" for e in window.context)
        blocks += f"RECAP (previous segment, context only — do not repeat):\n{recap_lines}\n\n"

    blocks += f"SPEECH:\n{speech_block}\n\nON SCREEN:\n{frames_block}"
    return blocks


def synthesize_window(window: Window, client=None) -> str:
    """
    Produce a grounded Markdown narrative for one window.

    Args:
        window: a Window from window_splitter (events: speech + frames; optional
                context: read-only recap of the previous window).
        client: an OpenAI-compatible client (injected for testing). If None, a
                real OpenAI client is created from settings.OPENAI_API_KEY.

    Returns:
        The narrative text for this segment. Empty string if the window has no
        usable content of its own (context alone does not trigger a narrative —
        we never describe only the recap).

    Raises:
        RuntimeError: if no client is given and the OpenAI key is missing.
    """
    # Nothing of this window's OWN to say → say nothing. Context alone is not
    # content (it belongs to the previous window and was already described there).
    if not window.events:
        return ""

    if client is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — cannot run synthesis. Add it to .env."
            )
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    user_content = _format_window_for_prompt(window)

    response = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,  # gpt-4o-mini: strong, cheap, good bilingual
        messages=[
            {"role": "system", "content": _SYNTHESIS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,  # deterministic, factual — lower creativity = less drift
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    text = (response.choices[0].message.content or "").strip() if response.choices else ""
    logger.info(
        "Synthesized window %d [%ds-%ds]: %d speech + %d frames (+%d recap) -> %d chars",
        window.index, window.start_seconds, window.end_seconds,
        len(window.speech_events), len(window.frame_events), len(window.context), len(text),
    )
    return text
