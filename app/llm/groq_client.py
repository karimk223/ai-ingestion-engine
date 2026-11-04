"""Write final answers using the configured Groq model."""
from openai import OpenAI, RateLimitError

from app.config import settings


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Send the answer rules and question/context to Groq; return final text."""
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required to generate answers through Groq.")

    try:
        with OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.ANSWER_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model=settings.ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning_effort="none",
                temperature=0,
            )
    except RateLimitError as exc:
        raise RuntimeError(
            "Groq's usage limit was reached. Please try again later."
        ) from exc

    if not response.choices:
        raise RuntimeError("Groq returned no answer.")

    choice = response.choices[0]
    answer = (choice.message.content or "").strip()
    if choice.finish_reason != "stop" or not answer:
        raise RuntimeError("Groq did not return a complete final answer.")

    return answer
