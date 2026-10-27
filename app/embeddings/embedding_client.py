import json
from pathlib import Path
from urllib.request import Request, urlopen


MODEL = "embeddinggemma:300m"
DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "


def embed_texts(texts):
    request = Request(
        "http://127.0.0.1:11434/api/embed",
        data=json.dumps({
            "model": MODEL,
            "input": texts,
            "truncate": False,
            "keep_alive": "0s",
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        embeddings = json.load(response)["embeddings"]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama returned an unexpected number of embeddings."
        )

    return embeddings


if __name__ == "__main__":
    input_path = Path("storage/chunks/sample.json")
    chunks = json.loads(input_path.read_text(encoding="utf-8"))

    documents = [DOCUMENT_PREFIX + chunk["text"] for chunk in chunks]
    vectors = embed_texts(documents)

    embedded_chunks = [
        {**chunk, "embedding": vector}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    output = {
        "model": MODEL,
        "document_prefix": DOCUMENT_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "chunks": embedded_chunks,
    }

    output_path = Path("storage/embeddings/sample.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(embedded_chunks)} embedded chunks to {output_path}.")
    print(f"Numbers per embedding: {len(vectors[0])}")
    print(f"First 5 numbers: {vectors[0][:5]}")