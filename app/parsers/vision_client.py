"""Describe video frames using the configured Gemini vision model."""
import base64

import cv2
from openai import OpenAI

from app.config import settings


_VISION_PROMPT = (
    "Describe this single video frame in at most 150 words. "
    "Read the main legible on-screen text, including headings, "
    "field labels and values, button labels, and important numbers. "
    "Preserve the original text's language, names, dates, amounts and units. "
    "Then briefly describe what is visible. "
    "Do not complete partial text or guess unreadable details. "
    "If the frame is blank or a plain transition, say 'blank frame'."
)


def describe_frame(frame_bgr) -> str:
    """Return the visible text and a description of one video frame."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for frame descriptions.")

    # Prepare the frame as a JPEG image.
    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return ""

    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    # Send the image and our instructions to Google.
    with OpenAI(
        api_key=settings.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=settings.VISION_TIMEOUT_SECONDS,
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=settings.VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            reasoning_effort="minimal",
            max_tokens=2000,
        )

    if not response.choices:
        return ""

    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(
            f"Gemini stopped because: {choice.finish_reason}"
        )

    return (choice.message.content or "").strip()
