# AI Ingestion Engine

A PDF and video ingestion project for retrieving information with references to its source. The current build connects PDF ingestion, multilingual semantic search and cited question answering, with optional Neo4j relationships to support GraphRAG answers.

## Available modules

- **PDF extraction:** selectable text, scanned pages and text inside embedded images, with page numbers and extraction-method labels.
- **Chunking:** sentence-based semantic splitting with an overlapping size-based fallback.
- **Embeddings:** local Ollama `embeddinggemma:300m`, with separate question/document prefixes and 768 dimensions checked before use.
- **Qdrant storage:** chunk text, embeddings and source metadata; similarity search and file filters.
- **PDF ingestion:** connects extraction, chunking, embeddings and storage while preserving page numbers.
- **Question search:** embeds a question and retrieves matching passages with source details.
- **Question answering:** sends retrieved passages to the configured Qwen model through Groq, returns answers with source records, and requests abstention when the documents do not contain the answer. Answers follow the question's language and preserve source amounts, currencies and conditions.
- **Connection extraction:** local Qwen produces subject/predicate/object candidates using a required JSON schema, name cleanup and duplicate removal. Invalid output is rejected before graph loading.
- **Multilingual graph preparation:** checks English, French and Arabic text locally, translates non-English sections to English through Groq, then sends the prepared text to local Qwen for relationship extraction. Original source text and references are preserved.
- **Question entities:** Groq produces an English version of the question first, then extracts English entity names for graph lookup.
- **Neo4j storage:** stores extracted connections with file provenance and supports node lookup and traversal across nearby relationships.
- **GraphRAG:** combines Qdrant passages with relevant Neo4j connections, preserves source labels and supports an optional file filter across both stores. Graph context can be switched off for passage-only answers.
- **HTTP API:** PDF upload, passage search, document lookup, ordinary RAG and GraphRAG, with an interactive test page at `/docs`.

Video processing, community analysis, entity resolution, meeting notes, n8n workflows and Obsidian exchange remain to be integrated and validated. The video modules currently include OpenAI clients for transcription, vision and synthesis; this path is not yet validated end to end. Building a graph from an ingested file currently runs through Python services; graph-assisted answering is available through the HTTP API.

## Local setup

Requires Python 3.12, Docker, Ollama and Tesseract. On macOS, install the OCR engine and additional language data with `brew install tesseract tesseract-lang`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull embeddinggemma:300m
ollama pull qwen3:4b
```

Existing local model downloads can be reused. Ollama runs Gemma embeddings and Qwen relationship extraction on your machine. Set `GROQ_API_KEY` in `.env` for final answers, question entity extraction and non-English graph-text translation. These requests use the configured Groq Qwen models; usage limits and availability depend on the account and model.

The example uses the Qdrant collection `content_chunks_gemma`. If changing from another embedding model, re-embed existing chunks into a separate collection before switching the application. Equal vector dimensions do not make different models' embeddings compatible.

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

With Ollama and Qdrant running and the Groq key configured, start the API from the repository root. Neo4j must also be running when using graph context:

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
| `POST` | `/graphrag/ask` | Answer with optional graph context; JSON body |
| `POST` | `/graphrag/ask-simple` | Same GraphRAG service; URL query parameters |
| `GET` | `/documents?name=` | List stored files, optionally filtering by part of a filename |

Both GraphRAG routes accept `use_graph` (default `true`), `top_k`, `hops` and an optional `file_id`. Setting `use_graph=false` skips graph retrieval on either route. A supplied file ID restricts both passage and relationship retrieval to that file. Build its graph using the Python service below before expecting graph evidence; PDF ingestion alone stores passages in Qdrant. Source labels help identify evidence, but retrieved facts and generated answers still need checking for relevance.

## Ingest and ask from Python

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

The service reads all stored source-text sections for the file from Qdrant. English sections pass through unchanged. French and Arabic sections are sent to Groq for English translation before local Qwen extracts connections. Long sections are split into translation requests, and original source text and references are retained. Completed translations are cached under the Git-ignored `storage/translations/` directory so retries can reuse them.

After preparation, the service extracts connections, removes duplicates and saves them to Neo4j. Translation failure stops the build before the existing graph is cleared. To inspect results without writing to Neo4j, call `extract_from_file(file_id)` from `app.services.extraction_service`; its report includes translation counts and the original/English text with source metadata. To reuse an existing extraction report, pass it as `extraction=extraction_report` when calling the builder; this intentionally skips fresh translation and extraction.

The local configuration uses `EXTRACTION_BATCH_CHARS=2000` and `EXTRACTION_TIMEOUT_SECONDS=180` to control text per extraction request and the time allowed for each request.
`TRANSLATION_BATCH_CHARS=2000` and `TRANSLATION_TIMEOUT_SECONDS=180` separately control translation requests. Translation and relationship extraction use model output, so their factual accuracy must still be checked against the source.

Graph loading updates relationships belonging to the selected file. The Neo4j browser at `http://localhost:7474` can show them:

```cypher
MATCH p=(:Entity)-[r:REL]->(:Entity)
WHERE r.file_id = 'YOUR_FILE_ID'
RETURN p;
```
