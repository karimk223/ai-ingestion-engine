# AI Ingestion Engine

A FastAPI service that makes PDFs and videos searchable and answers questions with source references. It combines OCR, speech transcription, multilingual semantic search and optional Neo4j graph context. The included n8n workflows automate uploads, graph building and meeting-minutes delivery, and provide a chat interface for the knowledge base. Standalone scripts support reviewing video documents before ingestion, processing folders of PDFs and videos, and exporting linked notes to Obsidian.

## Available modules

- **PDF extraction:** selectable text, scanned pages and text inside embedded images, with page numbers and extraction-method labels.
- **Chunking:** sentence-based semantic splitting with an overlapping size-based fallback.
- **Embeddings:** local Ollama `embeddinggemma:300m`, with separate question/document prefixes and 768 dimensions checked before use.
- **Qdrant storage:** chunk text, embeddings and source metadata; similarity search and file filters.
- **PDF ingestion:** connects extraction, chunking, embeddings and storage while preserving page numbers.
- **Video ingestion:** combines local Whisper transcripts with Gemini frame descriptions, synthesizes them into a document, then chunks, embeds and stores it in Qdrant. Video answers include filename references.
- **Question search:** embeds a question and retrieves matching passages with source details.
- **Question answering:** sends retrieved passages to the configured Qwen model through Groq, returns answers with source records, and requests abstention when the documents do not contain the answer. Answers follow the question's language and preserve source amounts, currencies and conditions.
- **Connection extraction:** local Qwen produces subject/predicate/object candidates using a required JSON schema, name cleanup and duplicate removal. Invalid output is rejected before graph loading.
- **Multilingual graph preparation:** checks English, French and Arabic text locally, translates non-English sections to English through Groq, then sends the prepared text to local Qwen for relationship extraction. Original source text and references are preserved.
- **Question entities:** Groq produces an English version of the question first, then extracts English entity names for graph lookup.
- **Neo4j storage:** stores extracted connections with file provenance and supports node lookup and traversal across nearby relationships.
- **GraphRAG:** combines Qdrant passages with relevant Neo4j connections, preserves source labels and supports an optional file filter across both stores. Graph context can be switched off for passage-only answers.
- **Entity resolution:** previews likely duplicate names across the graph and can merge them while preserving relationship direction and source file references.
- **Community summaries:** groups related entities with Neo4j Graph Data Science and uses Groq to summarize each eligible group from its relationships and relevant passages.
- **Meeting minutes:** generates structured minutes from a stored video, using a single pass or summarizing batches before combining them.
- **Reviewed video documents:** saves a synthesized document for human review, then indexes the reviewed text under the original video ID.
- **Folder ingestion:** processes PDFs and videos from one folder, builds each file's graph, and reports completed, skipped and failed files.
- **Obsidian export:** creates Markdown notes for stored PDFs and videos, with inline links and related-note suggestions across both file types.
- **n8n:** provides a form, a direct upload webhook, a chat workflow and a manual API health check.

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

Start Ollama if it is not already running. It hosts Gemma embeddings and local Qwen relationship extraction. Set `GROQ_API_KEY` in `.env` for answers, question entity extraction, graph-text translation, entity resolution, community summaries and meeting minutes.

For video ingestion, set `GEMINI_API_KEY` in `.env`. Gemini describes frames and combines the speech and visual evidence. Frame description failures fall back to local OCR. If Gemini returns HTTP 503 during synthesis, the API uses Groq for that synthesis request. Whisper runs on the CPU and downloads the selected model on first use. Audio decoding uses the FFmpeg binary bundled with `imageio-ffmpeg`.

The example configuration uses free-tier cloud API access. Provider quotas limit request frequency and input/output token budgets. For frequent ingestion or large files, configure accounts and models with sufficient limits; this may require a paid plan. Request sizes must still fit the selected model and account limits. Consult the provider's current account limits before running a large batch.

Cloud model selection is configured in `.env`:

| Provider | API key | Model settings |
|---|---|---|
| Groq | `GROQ_API_KEY` | `ANSWER_MODEL`, `QUESTION_ENTITY_MODEL`, `TRANSLATION_MODEL` |
| Gemini | `GEMINI_API_KEY` | `VISION_MODEL`, `SYNTHESIS_MODEL` |

Use compatible models with the required input capabilities and request parameters. Restart the API after changing `.env`; command-line scripts load configuration on each run. The current clients target Groq and Google's Gemini endpoints. Integrating OpenAI or another provider also requires adapting the relevant client endpoint, key selection and request parameters; changing model names or setting `OPENAI_API_KEY` alone does not switch providers.

The example uses the Qdrant collection `content_chunks_gemma`. If changing from another embedding model, re-embed existing chunks into a separate collection before switching the application. Equal vector dimensions do not make different models' embeddings compatible.

Start the local services. Neo4j is needed for graph features and the Obsidian exporter; n8n is needed for the workflows.

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n:latest
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -v neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/CHANGE_ME \
  -e 'NEO4J_PLUGINS=["graph-data-science"]' neo4j:5
```

Replace `CHANGE_ME` with your local database password and set the same value as `NEO4J_PASSWORD` in `.env`. The example configuration supplies the local addresses and model settings. Keep credentials in `.env`, which is excluded from Git.

For existing containers, use `docker start qdrant n8n neo4j` instead of creating them again.

## Inspect text extraction from a PDF

From the repository root, replace `path/to/your-document.pdf` with the path to a PDF you want to inspect.

```bash
python - "path/to/your-document.pdf" <<'PY'
import sys
from pathlib import Path

from app.parsers.pdf_parser import extract_pdf_pages

path = Path(sys.argv[1])
for number, text, method in extract_pdf_pages(path.read_bytes()):
    print(f"\n--- Page {number} | {method} ---")
    print(text.strip())
PY
```

The reported method tells you where the text came from. `pdf_text` means text was read directly from the page, `ocr` means it was recognized from the scanned page, and `pdf_text+ocr` combines page text with text recognized from embedded images. `OCR_LANGUAGES` defaults to `eng+fra+ara` for English, French and Arabic. Use `tesseract --list-langs` to check the installed language data.

## Video ingestion

Upload a video through `POST /ingest/video` in `/docs`. The endpoint accepts `.mp4`, `.mov`, `.avi`, `.mkv` and `.webm` files up to 200 MB and returns its `file_id`, retained frame count, transcript segment count and stored chunk count.

For a recording that switches languages, set `mixed_language=true` in Swagger or call `POST /ingest/video?mixed_language=true`. FFmpeg decodes the audio and detects pauses of at least one second. The service splits the audio at eligible pauses and uses Whisper `small` to detect the language and transcribe each part separately. The regular upload uses the configured `WHISPER_MODEL`, which defaults to `base`.

The example configuration samples frames every five seconds and paces model calls with `LLM_CALL_DELAY_SECONDS`. With `DESCRIBE_FRAMES=false`, frame processing uses local OCR to read visible text. Synthesis groups the frame descriptions and timestamped speech into windows and combines them into one Markdown document before indexing. The resulting text chunks cite the video filename. Transcription failures and vision fallbacks are logged.

## HTTP API

With Ollama and Qdrant running and the Groq key configured, start the API from the repository root. Neo4j must also be running when using graph context.

```bash
python -m uvicorn app.main:app --reload
```

Open [Swagger UI](http://localhost:8000/docs) to upload a PDF with `POST /ingest/pdf` or a video with `POST /ingest/video`, then ask questions with `POST /ask`. Include the returned `file_id` to restrict retrieval to that file. `/ask` accepts a JSON body and uploads use multipart form data. Restart the API after changing `.env` settings.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Check the API is running |
| `POST` | `/ingest/pdf` | Extract, chunk, embed and store a PDF |
| `POST` | `/ingest/video` | Process speech and frames, synthesize, chunk, embed and store a video |
| `POST` | `/search` | Retrieve matching passages and source metadata |
| `POST` | `/ask` | Answer from retrieved passages with source records |
| `POST` | `/extract/{file_id}` | Preview extracted connections without writing to Neo4j |
| `POST` | `/graph/build/{file_id}` | Extract connections for a stored file and save them to Neo4j |
| `GET` | `/graph/stats` | Count nodes and relationships across the whole graph |
| `POST` | `/graph/resolve-entities?apply=false` | Preview suggested duplicate merges across the graph |
| `POST` | `/graph/communities?min_size=2` | Detect communities and summarize groups that meet the size threshold |
| `POST` | `/graphrag/ask` | Answer with optional graph context; JSON body |
| `POST` | `/graphrag/ask-simple` | Same GraphRAG service; URL query parameters |
| `GET` | `/documents?name=` | List stored files, optionally filtering by part of a filename |
| `POST` | `/minutes/{file_id}` | Generate minutes for an ingested video by ID |
| `POST` | `/minutes/by-name?name=` | Find one ingested video by filename and generate its minutes |

Both GraphRAG routes accept `use_graph` (default `true`), `top_k`, `hops` and an optional `file_id`. Setting `use_graph=false` skips graph retrieval on either route. A supplied file ID restricts both passage and relationship retrieval to that file. Build its graph with `POST /graph/build/{file_id}` or the Python service below before expecting graph evidence; PDF and video ingestion store passages in Qdrant, while graph extraction and storage are separate steps. Source labels help identify evidence, but retrieved facts and generated answers still need checking for relevance.

## Ingest and ask from Python

With Ollama and Qdrant running, pass your PDF path and a question about that document. The answer uses the upload's returned file ID to find passages from that PDF.

```bash
python - "path/to/your-document.pdf" "What are the main recommendations?" <<'PY'
import sys
from pathlib import Path

from app.services.answer_service import answer_question
from app.services.ingestion_service import ingest_pdf

pdf = Path(sys.argv[1])
report = ingest_pdf(pdf.read_bytes(), pdf.name)
print(f"{report.file_id}: {report.pages_processed} pages, {report.chunks_created} chunks")

result = answer_question(
    sys.argv[2],
    top_k=3,
    file_id=report.file_id,
    source_type="pdf",
)
print(result.answer)
print("Retrieved source candidates:")
for source in result.sources:
    print(f"- {source.label}")
PY
```

## Extract connections and build a graph

Use the `file_id` returned by PDF or video ingestion. In `/docs`, `POST /extract/{file_id}` provides an optional preview without saving to Neo4j. `POST /graph/build/{file_id}` performs extraction and saves the result; a preview is not required. `GET /graph/stats` shows whole-graph totals.

With Qdrant, Ollama and Neo4j running and the Groq key configured, the same build is available from Python:

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

### Resolve duplicates and summarize communities

Use `POST /graph/resolve-entities` to review proposed merges and the reason for each one. The default `apply=false` previews suggestions across the whole graph. An `apply=true` request generates its own set of suggestions and applies them, so its pairs can differ from a previous preview.

`POST /graph/communities` runs Louvain on a temporary in-memory graph projection. It writes community IDs to the Neo4j entity nodes and drops the projection afterward. Groq receives each qualifying group's internal relationships and relevant passages from Qdrant when available. `min_size` controls the smallest group to summarize. A higher value can reduce the number of Groq calls when trying the endpoint on a populated graph.

## Generate meeting minutes

After ingesting a recording, call `POST /minutes/{file_id}` or `POST /minutes/by-name?name=your-recording.mp4`. Name matching is case-insensitive and accepts a partial filename that matches a single video. Both endpoints read the stored video content from Qdrant and use Groq to write the minutes. The prompts instruct the model to record demo activities under Key Points and reserve Decisions and Action Items for explicit commitments.

`MOM_BATCH_CHARS` sets the character budget used to group the stored text. When the text fits in one batch, the response reports `single-pass`. With multiple batches, `map-reduce` writes partial minutes for each batch and combines them in a final call.

## Review a video document before ingestion

The two helper scripts separate document generation from indexing. Run them from the repository root in an activated virtual environment.

First, generate a Markdown document from a video:

```bash
python synthesize_video.py "path/to/video.mp4" "reviewed_document.md"
```

The script combines frame descriptions and the speech transcript, saves the document, and stops before writing to Qdrant. Open the Markdown file and review or edit its contents. If no usable speech or on-screen content is available, generation stops with an explanatory error. When only one source is available, the script reports whether it is using speech or on-screen content alone.

Index the reviewed document using the original video file:

```bash
python ingest_document.py "reviewed_document.md" "path/to/video.mp4"
```

This stores the reviewed text without repeating frame processing or synthesis. The original video's contents determine its file ID, matching normal video ingestion. Re-indexing updates that video's chunks and removes any surplus chunks left by an older, longer document. Empty documents are rejected before storage. Rebuild the file's graph afterward using the returned file ID, as described under **Extract connections and build a graph**.

## Ingest a folder of PDFs and videos

The folder script runs ingestion and graph building automatically for each supported file. It requires Ollama, Qdrant and Neo4j, plus the cloud credentials required by the selected processing steps. It calls the application services directly, so a running FastAPI server is not required.

```bash
python ingest_folder.py "path/to/folder"
```

The script reads the folder's top level and accepts `.pdf`, `.mp4`, `.mov`, `.avi`, `.mkv` and `.webm` files, including uppercase extensions. Example input:

```text
input/
    guide.pdf
    demo.mp4
    demo_document.md
    training.mov
```

PDFs use the PDF extraction, OCR, chunking and embedding pipeline. For each video, the script looks for a companion Markdown file named `<video_stem>_document.md` in the same folder. If present, that document is ingested, preserving any edits. Otherwise, the script generates it and immediately ingests it, without pausing for review. Existing Qdrant data does not prevent a supplied document from updating that video's indexed text. Use distinct video basenames within a folder so each video has its own companion document.

Graph building follows ingestion for each non-skipped file. Zero-byte files and PDFs with no readable text are skipped with a reason. Videos with no usable speech or on-screen content, or documents that produce no chunks, are also skipped. Other processing errors are reported per file while the batch continues. A summary reports completed, skipped and failed counts; exit status is `0` when no files failed and `1` when at least one did. The script waits five seconds between files.

Indexing and graph building are separate stages. After updating indexed text, rebuild its graph to reflect that text; rerunning the folder script performs both stages. To review every video document before indexing, generate and review the documents with the single-video helper first, then run the folder importer with those companion files present. There is no separate folder-wide preparation command.

## Export PDF and video notes to Obsidian

With Qdrant and Neo4j running and files already ingested, run this command from the repository root in the activated virtual environment:

```bash
python obsidian_export.py
```

The exporter creates one Markdown note per stored PDF or video in `obsidian_vault/`. It reads the saved text from Qdrant, restores chunk order, and uses the extracted relationships in Neo4j to choose links between notes. It includes PDF-to-PDF, PDF-to-video and video-to-video links when eligible matches exist. Files without extracted relationships still receive a note. Build each file's graph first to make its entities available for linking.

Each note contains the saved PDF text or synthesized video document. The exporter adds at most eight inline links, with one new link per selected entity, and up to three related notes under **See also**. It preserves existing Markdown links and code blocks when inserting links. Shared entity names and connecting relationships guide the suggestions; the links are candidates for navigation, rather than proof that two files discuss the same subject. PDF text is exported as text, without recreating the original page layout.

To choose a different output folder, pass its path:

```bash
python obsidian_export.py /path/to/your/vault
```

Open Obsidian, choose **Open folder as vault**, and select the generated folder. Click the inline links or **See also** entries to navigate between notes. Run **Graph view: Open graph view** from Obsidian's command palette to see notes as dots and note links as lines. Notes without links appear separately.

Re-running the export overwrites the corresponding Markdown notes. Edits made in Obsidian stay in the note files and are not imported into Qdrant or Neo4j. The default `obsidian_vault/` folder is excluded from Git.

## n8n workflows

With FastAPI running, open [n8n](http://localhost:5678) and use the workflow menu's **Import from File** option for the exports below. The HTTP Request nodes use Docker Desktop's `host.docker.internal` to reach the host. Update the node URLs if your n8n container connects to FastAPI or the email webhook through a different address. [n8n workflow import guide](https://docs.n8n.io/build/manage-workflows/export-and-import/#from-the-editor-ui-menu)

| Workflow | What it does |
|---|---|
| [API Health Check](n8n_workflows/api_health_check.json) | Manually calls `/health` to check that n8n can reach FastAPI |
| [Upload & Ingest](n8n_workflows/upload_ingest_minutes_agent_workflow.json) | Uploads PDFs or videos from a form and builds each file's graph. Videos also go through a separate branch that generates minutes and emails them. The chat agent can search passages, ask GraphRAG questions, get community summaries and email minutes for an already ingested video. |
| [Direct Upload Webhook](n8n_workflows/ingest_webhook.json) | Takes a multipart `file` from an HTTP client, forwards it to the PDF or video ingestion route and returns the ingestion report |

Run **API Health Check** first and look for `{"status":"ok"}` in the HTTP Request output.

In **Upload & Ingest**, configure the Groq Chat Model credential and select your Gmail OAuth2 credential on both Gmail nodes. Replace `you@example.com` on both nodes with your intended recipient. Enable the Gmail API in Google Cloud and enter the redirect URI shown by n8n when creating the OAuth client. For a credential used only to send minutes, set the custom scope to `https://www.googleapis.com/auth/gmail.send` and leave the optional Sender Name blank. [n8n Google OAuth2 setup](https://docs.n8n.io/integrations/builtin/credentials/google/oauth-single-service/)

To try a form upload, run the workflow in the editor and submit a PDF through the test form. Open **Ingest file** to see its `file_id` and **Build graph** to see the graph report. The chat email tool uses the production `/webhook/email-minutes` URL, which is registered when you publish the workflow. [n8n webhook URLs](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/#webhook-urls)

To try **Direct Upload Webhook**, click **Listen for test event** on its Webhook node, then run this command with your chosen file. The response includes the `file_id` for later search or graph building. [n8n webhook testing](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/#webhook-urls)

```bash
curl --fail-with-body \
  -F 'file=@path/to/your-document.pdf' \
  'http://localhost:5678/webhook-test/ingest'
```

## Tests and runtime storage

After installing dependencies and configuring `.env`, run the automated tests from the repository root.

```bash
python -m unittest discover -s tests
```

The tests cover translation routing, cache reuse, and the order of steps in the graph pipeline. They replace external calls with test doubles, so they can run without starting the services.

The app creates its local `storage/` subdirectories as needed. Video uploads use `storage/uploads/` for temporary files, which are removed after processing. Completed translations are cached under `storage/translations/`. Qdrant and Neo4j keep their database contents in the Docker volumes from the setup commands.
