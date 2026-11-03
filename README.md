# AI Ingestion Engine

A PDF and video ingestion project for retrieving information with references to its source. The current build connects PDF ingestion, semantic search and local question answering with page references, plus connection extraction and Neo4j graph storage.

## Available modules

- **PDF extraction:** selectable text, scanned pages and text inside embedded images, with page numbers and extraction-method labels.
- **Chunking:** sentence-based semantic splitting with an overlapping size-based fallback.
- **Embeddings:** local Ollama `nomic-embed-text`, with 768 dimensions checked before use.
- **Qdrant storage:** chunk text, embeddings and source metadata; similarity search and file filters.
- **PDF ingestion:** connects extraction, chunking, embeddings and storage while preserving page numbers.
- **Question search:** embeds a question and retrieves matching passages with source details.
- **Local question answering:** assembles retrieved context for Ollama `qwen3:4b`, returns the final answer with source records, and requests abstention when the documents do not contain the answer.
- **Connection extraction:** local Qwen produces subject/predicate/object candidates using a required JSON schema, name cleanup and duplicate removal. Invalid output is rejected before graph loading.
- **Multilingual graph preparation:** checks English, French and Arabic text locally, translates non-English sections to English through Groq, then sends the prepared text to local Qwen for relationship extraction. Original source text and references are preserved.
- **Question entities:** Groq produces an English version of the question first, then extracts English entity names for graph lookup.
- **Neo4j storage:** stores extracted connections with file provenance and supports node lookup and traversal across nearby relationships.
- **HTTP API:** PDF upload, passage search, document lookup and question answering, with an interactive test page at `/docs`.

Video processing, graph-assisted answers, community analysis, entity resolution, meeting notes, n8n workflows and Obsidian exchange remain to be integrated and validated. The video modules currently include OpenAI clients for transcription, vision and synthesis; they are separate from the local PDF path. Graph extraction and storage currently run through Python services rather than HTTP routes.

## Local setup

Requires Python 3.12, Docker, Ollama and Tesseract. On macOS, install the OCR engine and additional language data with `brew install tesseract tesseract-lang`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

The answer model download is approximately 2.5 GB. Existing local copies can be reused. The PDF path uses local embeddings and local answer generation and does not require a cloud account.

For graph-text translation and question entity extraction, set `GROQ_API_KEY` in `.env`. Both use the configured Groq Qwen model; relationship extraction and ordinary `/ask` answers still use local Ollama. Groq's free-plan limits apply, and availability depends on the account and model.

Start the local services:

```bash
printf '%s\n' 'NEO4J_PLUGINS=["graph-data-science"]' > neo4j.env
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n:latest
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -v neo4j_data:/data -e NEO4J_AUTH=neo4j/CHANGE_ME --env-file neo4j.env neo4j:5
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

The sample exercises `pdf_text`, `ocr` and `pdf_text+ocr`. `OCR_LANGUAGES` defaults to `eng+fra+ara` for English, French and Arabic. Confirm these language files are installed with `tesseract --list-langs`; both whole-page and embedded-image OCR use this setting. Recognition quality depends on the scan and page layout.

## HTTP API

With Ollama and Qdrant running, start the API from the repository root:

```bash
python -m uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` to upload a PDF with `POST /ingest/pdf`, then ask questions with `POST /ask`. Include the returned `file_id` to restrict a question to that document. Questions accept a JSON body; PDF uploads use a file form.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Check the API is running |
| `POST` | `/ingest/pdf` | Extract, chunk, embed and store a PDF |
| `POST` | `/search` | Retrieve matching passages and source metadata |
| `POST` | `/ask` | Answer from retrieved passages with source records |
| `GET` | `/documents?name=` | List stored files, optionally filtering by part of a filename |

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

## Extract connections and build a graph

With Qdrant, Ollama and Neo4j running, use the `file_id` returned by PDF ingestion:

```python
from app.services.graph_service import build_graph_for_file

report = build_graph_for_file("YOUR_FILE_ID")
print(report)
```

The service reads all stored source-text sections for the file from Qdrant. English sections pass through unchanged. French and Arabic sections are translated completely into English before local Qwen extracts connections. Long sections are split into translation requests; no source text is truncated. Completed translations are cached under the Git-ignored `storage/translations/` directory so retries can reuse them.

After preparation, the service extracts connections, removes duplicates and saves them to Neo4j. Translation failure stops the build before the existing graph is cleared. To inspect results without writing to Neo4j, call `extract_from_file(file_id)` from `app.services.extraction_service`; its report includes translation counts and the original/English text with source metadata. To reuse an existing extraction report, pass it as `extraction=extraction_report` when calling the builder; this intentionally skips fresh translation and extraction.

The local configuration uses `EXTRACTION_BATCH_CHARS=2000` and `EXTRACTION_TIMEOUT_SECONDS=180` to control text per extraction request and the time allowed for each request.
`TRANSLATION_BATCH_CHARS=2000` and `TRANSLATION_TIMEOUT_SECONDS=180` separately control translation requests. Translation and relationship extraction use model output, so their factual accuracy must still be checked against the source.

Graph loading updates relationships belonging to the selected file. The Neo4j browser at `http://localhost:7474` can show them:

```cypher
MATCH p=(:Entity)-[r:REL]->(:Entity)
WHERE r.file_id = 'YOUR_FILE_ID'
RETURN p;
```
