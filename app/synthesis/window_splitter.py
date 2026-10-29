"""
Window splitter — Step 1 of the video-synthesis pipeline.

Takes the two raw streams a video produces:
  - frames:     list of (timestamp_seconds, frame_number, description)
  - transcript: list of (start_seconds, text)

and merges them into time-ordered WINDOWS. Each window collects everything (speech
+ frame descriptions) that occurred within a span of the video, small enough to fit
comfortably in an LLM's context so a synthesis agent can attend to all of it.

This is pure, deterministic logic — no LLM, no I/O — so it can be tested with hard
assertions. The window-boundary policy is a PARAMETER (size budget), so whatever
window size is decided later is just an argument, not a rewrite.

Design notes:
- We interleave both streams by timestamp into a single ordered list of "events,"
  each tagged with its kind ("speech" or "frame"), then pack events into windows
  until the next event would exceed the size budget.
- Size is measured in characters (a cheap, stable proxy for tokens — ~4 chars/token).
- An oversized single event still gets its own window rather than being split.

Overlap (context between windows):
  A window is cut by a size budget, not by meaning, so a topic can split across a
  boundary — leaving the next window's synthesis agent without the context the
  previous window established. To soften those seams, each window (after the first)
  carries a small READ-ONLY tail of the previous window as `context` events. The
  synthesis agent is told to use this recap to understand what follows but NOT to
  re-describe it. Context is stored SEPARATELY from `events`, so it never affects
  the "nothing dropped / nothing duplicated / size budget" guarantees — it is
  purely additive reference material.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowEvent:
    """One thing that happened at a point in the video."""
    timestamp: int            # whole-second offset in the video
    kind: str                 # "speech" or "frame"
    text: str                 # the transcript text or the frame description
    frame_number: int | None = None  # set only for frame events


@dataclass
class Window:
    """A time-bounded slice of the video: everything that happened within it."""
    index: int                       # 0-based window position
    start_seconds: int               # timestamp of the first event in the window
    end_seconds: int                 # timestamp of the last event in the window
    events: list[WindowEvent] = field(default_factory=list)
    # Read-only tail of the PREVIOUS window, for continuity. Not part of this
    # window's own content — the synthesis agent uses it as recap, not new material.
    context: list[WindowEvent] = field(default_factory=list)

    @property
    def char_size(self) -> int:
        """Total characters of text in this window's OWN events (the budget metric)."""
        return sum(len(e.text) for e in self.events)

    @property
    def speech_events(self) -> list[WindowEvent]:
        return [e for e in self.events if e.kind == "speech"]

    @property
    def frame_events(self) -> list[WindowEvent]:
        return [e for e in self.events if e.kind == "frame"]


def _merge_streams(
    frames: list[tuple[int, int, str]],
    transcript: list[tuple[int, str]],
) -> list[WindowEvent]:
    """
    Interleave frames and transcript into a single timestamp-ordered event list.

    Empty/whitespace-only texts are dropped. Ties (same timestamp) put speech
    before the frame, since the spoken content is usually the higher-value signal.
    """
    events: list[WindowEvent] = []

    for ts, text in transcript:
        text = (text or "").strip()
        if text:
            events.append(WindowEvent(timestamp=int(ts), kind="speech", text=text))

    for ts, frame_no, text in frames:
        text = (text or "").strip()
        if text and text.lower() != "blank frame":  # skip the vision model's blanks
            events.append(
                WindowEvent(timestamp=int(ts), kind="frame", text=text, frame_number=frame_no)
            )

    events.sort(key=lambda e: (e.timestamp, 0 if e.kind == "speech" else 1))
    return events


def split_into_windows(
    frames: list[tuple[int, int, str]],
    transcript: list[tuple[int, str]],
    max_chars: int = 4000,
    overlap_events: int = 2,
) -> list[Window]:
    """
    Merge the two streams and pack them into time-ordered windows under a size budget,
    with a small read-only overlap carried from each window into the next.

    Args:
        frames:         (timestamp_seconds, frame_number, description) tuples.
        transcript:     (start_seconds, text) tuples.
        max_chars:      soft cap on characters per window (THE boundary policy).
        overlap_events: how many trailing events of a window to carry into the next
                        window as read-only `context`. 0 disables overlap. Default 2.

    Returns:
        A list of Window objects, in time order. Empty input -> empty list.

    Guarantees (locked by tests):
        - Every non-empty event appears in exactly one window's `events`
          (nothing dropped, nothing duplicated). Overlap is COPIED into the next
          window's separate `context` list, so it does not violate this.
        - Events within and across windows are in non-decreasing timestamp order.
        - No window's OWN events exceed max_chars unless it holds a single
          oversized event.
        - start_seconds/end_seconds bound each window's own events.
    """
    events = _merge_streams(frames, transcript)
    if not events:
        return []

    windows: list[Window] = []
    current: list[WindowEvent] = []
    current_size = 0

    for ev in events:
        ev_len = len(ev.text)
        if current and current_size + ev_len > max_chars:
            windows.append(_finalize(len(windows), current))
            current = [ev]
            current_size = ev_len
        else:
            current.append(ev)
            current_size += ev_len

    if current:
        windows.append(_finalize(len(windows), current))

    # Add read-only overlap: each window (after the first) gets the last
    # `overlap_events` events of the previous window as context. This is a COPY
    # for reference — the events still "belong" to the previous window.
    if overlap_events > 0:
        for i in range(1, len(windows)):
            prev_tail = windows[i - 1].events[-overlap_events:]
            windows[i].context = list(prev_tail)

    return windows


def _finalize(index: int, events: list[WindowEvent]) -> Window:
    """Build a Window from an accumulated event list, computing its time bounds."""
    return Window(
        index=index,
        start_seconds=events[0].timestamp,
        end_seconds=events[-1].timestamp,
        events=list(events),
    )
