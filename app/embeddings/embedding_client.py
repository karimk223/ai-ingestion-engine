"""Create embeddings for questions and document chunks."""

import httpx

from app.config import settings


_BATCH_SIZE = 4


def _prepare_text(text: str, *, is_query: bool) -> str:
    if settings.EMBEDDING_MODEL.split(":")[0] == "embeddinggemma":
        if is_query:
            return "task: search result | query: " + text

        return "title: none | text: " + text

    return text


def _request_embeddings(
    client: httpx.Client,
    texts: list[str],
) -> list[list[float]]:
    response = client.post(
        f"{settings.OLLAMA_BASE_URL}/api/embed",
        json={
            "model": settings.EMBEDDING_MODEL,
            "input": texts,
            "truncate": False,
        },
    )
    response.raise_for_status()

    vectors = response.json()["embeddings"]

    if len(vectors) != len(texts):
        raise RuntimeError(
            "Ollama returned an unexpected number of embeddings."
        )

    for vector in vectors:
        if len(vector) != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected {settings.EMBEDDING_DIM} numbers, "
                f"received {len(vector)}."
            )

    return vectors


def embed_text(text: str) -> list[float]:
    """Create an embedding for one search question."""
    prepared = _prepare_text(text, is_query=True)

    with httpx.Client(timeout=180.0) as client:
        return _request_embeddings(client, [prepared])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Create embeddings for document text, keeping the same order."""
    if not texts:
        return []

    prepared = [
        _prepare_text(text, is_query=False)
        for text in texts
    ]
    vectors: list[list[float]] = []

    with httpx.Client(timeout=180.0) as client:
        for start in range(0, len(prepared), _BATCH_SIZE):
            batch = prepared[start:start + _BATCH_SIZE]
            vectors.extend(_request_embeddings(client, batch))

    return vectors