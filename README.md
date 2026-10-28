# AI Ingestion Engine

A PDF and video ingestion project for retrieving information with references to its source. The current build provides the PDF processing, chunking, embedding and vector-storage modules.

## Available modules

- **PDF extraction:** selectable text, scanned pages and text inside embedded images, with page numbers and extraction-method labels.
- **Chunking:** sentence-based semantic splitting with an overlapping size-based fallback.
- **Embeddings:** local Ollama `nomic-embed-text`, with 768 dimensions checked before use.
- **Qdrant storage:** chunk text, embeddings and source metadata; similarity search and file filters.

Video processing, the HTTP API, answer generation, graph retrieval, meeting notes and n8n workflows remain to be integrated. The complete ingestion-to-answer pipeline is not yet available.

## Local setup

Requires Python 3.12, Docker, Ollama and Tesseract. On macOS, install the OCR engine with `brew install tesseract`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
```

Start the local services:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n:latest
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/CHANGE_ME --env-file neo4j.env neo4j:5
```

Replace `CHANGE_ME` with your local database password and set the same value as `NEO4J_PASSWORD` in `.env`. The example configuration supplies the local addresses and model settings. Keep credentials in `.env`, which is excluded from Git.

For existing containers, use `docker start qdrant n8n neo4j` instead of creating them again.

## PDF example

`examples/parser_sample.pdf` contains three fictional pages: selectable text, a scanned page and a text page with an embedded image. Run from the repository root:

```bash
python - <<'PY'
from pathlib import Path
from app.parsers.pdf_parser import extract_pdf_pages

path = Path("examples/parser_sample.pdf")
for number, text, method in extract_pdf_pages(path.read_bytes()):
    print(f"\n--- Page {number} | {method} ---")
    print(text.strip())
PY
```

The sample exercises `pdf_text`, `ocr` and `pdf_text+ocr`. Tesseract's base installation uses English; additional OCR languages require language data and configuration.

Local checks have confirmed the three PDF extraction paths, 768-dimensional embeddings and a Qdrant collection configured with cosine comparison. Semantic boundary evaluation and end-to-end integration are still in progress.
