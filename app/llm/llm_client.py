"""
LLM client — wraps Ollama's /api/chat endpoint.

One job: given a system prompt + user prompt, return the assistant's text answer.

Works with both local models (via localhost:11434) and Ollama Cloud models
(model names ending in -cloud). Same endpoint, same API — routing happens
inside Ollama based on the model name.

Reasoning may arrive in a separate `thinking` field or inline before a closing
`</think>` marker. Return only the final answer, without the reasoning section.
"""
import httpx
from app.config import settings


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    json_schema: dict | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """
    Ask the LLM to respond to a user_prompt under a given system_prompt.

    json_mode requests JSON text; json_schema supplies its required structure.
    timeout_seconds overrides the default wait for this request only.

    Returns:
        The assistant's final text (thinking output stripped).

    Raises:
        RuntimeError: if the response is missing the expected content field.
        httpx.HTTPError: if the Ollama call itself fails.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    if settings.LLM_MODEL.startswith("qwen3:"):
        user_prompt = f"{user_prompt}\n/no_think"

    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
    }

    if json_schema is not None:
        payload["format"] = json_schema
    elif json_mode:
        payload["format"] = "json"

    if json_mode or json_schema is not None:
        payload["options"] = {"temperature": 0}

    timeout = settings.LLM_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"Ollama returned HTTP {response.status_code}: {response.text}",
                request=exc.request,
                response=exc.response,
            ) from exc
        data = response.json()

    message = data.get("message", {})
    content = message.get("content", "") or ""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    if "<think>" in content or not content.strip():
        raise RuntimeError("LLM did not return a complete final answer.")
    return content.strip()
