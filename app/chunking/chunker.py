import json
from pathlib import Path
import re


def chunk_pages(pages, chunk_size=100, overlap=20):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Use chunk_size > 0 and 0 <= overlap < chunk_size."
        )

    chunks = []

    for page in pages:
        paragraphs = re.split(r"\n\s*\n", page["text"].strip())
        chunk_number = 0

        for paragraph in paragraphs:
            words = paragraph.split()
            step = chunk_size - overlap

            for start in range(0, len(words), step):
                chunk_number += 1

                chunks.append({
                    "file_name": page["file_name"],
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": " ".join(words[start:start + chunk_size]),
                })

                if start + chunk_size >= len(words):
                    break

    return chunks


if __name__ == "__main__":
    input_path = Path("storage/extracted/sample.json")
    pages = json.loads(input_path.read_text(encoding="utf-8"))

    chunks = chunk_pages(pages)

    output_path = Path("storage/chunks/sample.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(chunks)} chunks to {output_path}.")