import json
import re
from pathlib import Path


STOP_WORDS = {
    "a", "an", "the", "is", "are", "what", "how", "do", "does",
    "i", "to", "of", "in", "for", "my", "can", "and", "on",
}


def tokenize(text):
    words = set(re.findall(r"\w+", text.casefold()))
    return words - STOP_WORDS


def search_chunks(question, chunks, top_k=3):
    query_words = tokenize(question)
    results = []

    for chunk in chunks:
        matching_words = query_words & tokenize(chunk["text"])

        if matching_words:
            results.append({
                **chunk,
                "score": len(matching_words),
            })

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    input_path = Path("storage/chunks/sample.json")
    chunks = json.loads(input_path.read_text(encoding="utf-8"))

    question = input("Ask a question: ").strip()
    results = search_chunks(question, chunks)

    if not results:
        print("No matching passages found.")

    for result in results:
        print(
            f"\n{result['file_name']} | Page {result['page_number']} "
            f"| Chunk {result['chunk_number']}"
        )
        print(f"Matching words: {result['score']}")
        print(result["text"])