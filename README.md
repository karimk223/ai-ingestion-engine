# AI Ingestion Engine

A PDF and video ingestion project for retrieving information with references to its source. The current build connects PDF ingestion, semantic search and local question answering with page references.

## Available modules

- **PDF extraction:** selectable text, scanned pages and text inside embedded images, with page numbers and extraction-method labels.
- **Chunking:** sentence-based semantic splitting with an overlapping size-based fallback.
- **Embeddings:** local Ollama `nomic-embed-text`, with 768 dimensions checked before use.
- **Qdrant storage:** chunk text, embeddings and source metadata; similarity search and file filters.
- **PDF ingestion:** connects extraction, chunking, embeddings and storage while preserving page numbers.
- **Question search:** embeds a question and retrieves matching passages with source details.
- **Local question answering:** assembles retrieved context for Ollama `qwen3:4b`, returns the final answer with source records, and requests abstention when the documents do not contain the answer.

Video processing, the HTTP API, graph retrieval, meeting notes and n8n workflows remain to be integrated and validated. The video modules currently include OpenAI clients for transcription, vision and synthesis; they are separate from the local PDF path.

## Local setup

Requires Python 3.12, Docker, Ollama and Tesseract. On macOS, install the OCR engine with `brew install tesseract`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

The answer model download is approximately 2.5 GB. Existing local copies can be reused. The PDF path uses local embeddings and local answer generation and does not require a cloud account.

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

## Ingest and ask locally

With Ollama and Qdrant running, ingest the sample PDF from the repository root:

```bash
python - <<'PY'
from pathlib import Path
from app.services.ingestion_service import ingest_pdf

pdf = Path("examples/parser_sample.pdf")
report = ingest_pdf(pdf.read_bytes(), pdf.name)
print(f"Processed {report.pages_processed} pages; stored {report.chunks_created} chunks.")
PY
```

Ask a question about the ingested content:

```bash
python - <<'PY'
from app.services.answer_service import answer_question

result = answer_question(
    "How long do I have to return a product?",
    top_k=3,
    source_type="pdf",
)
print(result.answer)
print("Retrieved source candidates:")
for source in result.sources:
    print(f"- {source.label}")
PY
```

The local Qwen client requests non-thinking output and removes inline reasoning sections from the displayed answer. If the model provides no complete final answer, the client raises an error instead of returning that reasoning text.

## Validation and scope

Checks on the supplied English fixture confirmed the three PDF extraction paths, 768-dimensional embeddings, ingestion of six chunks and readback from Qdrant with page metadata. The local answer flow returned the correct returns deadline and OCR-extracted support email with page references, and declined a question about an unspecified CEO. These are small fixture checks; broader document and language evaluation remains in progress. Returned source records describe retrieved candidates and do not certify that every passage was used in the answer.
