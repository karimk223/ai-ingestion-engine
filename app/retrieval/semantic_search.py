import json
from pathlib import Path

from app.embeddings.embedding_client import (
    MODEL,
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    embed_texts,
)


def search_chunks(question, data, top_k=3):
    if not question.strip() or not data["chunks"]:
        return []

    if (
        data["model"] != MODEL
        or data["document_prefix"] != DOCUMENT_PREFIX
        or data["query_prefix"] != QUERY_PREFIX
    ):
        raise ValueError(
            "Saved embeddings use a different model or input format."
        )

    query_vector = embed_texts([QUERY_PREFIX + question])[0]
    results = []

    for chunk in data["chunks"]:
        score = sum(
            a * b
            for a, b in zip(
                query_vector, chunk["embedding"], strict=True
            )
        )
        results.append({**chunk, "score": score})

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    input_path = Path("storage/embeddings/sample.json")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    question = input("Ask a question: ").strip()
    if not question:
        raise SystemExit("Please enter a question.")

    results = search_chunks(question, data)

    for result in results:
        print(
            f"\n{result['file_name']} | Page {result['page_number']} "
            f"| Chunk {result['chunk_number']}"
        )
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])