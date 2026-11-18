import sys
import time
from pathlib import Path
from app.services.ingestion_service import ingest_pdf

from synthesize_video import synthesize_video
from ingest_document import ingest_document
from app.services.graph_service import build_graph_for_file

def process_video(video_path: Path) -> dict:
    print(f"Processing: {video_path.name}")

    if video_path.stat().st_size == 0:
        print(f"Skipped {video_path.name}: empty file.")
        return {"file_name": video_path.name, "skipped": True}

    document_path = video_path.with_name(
        f"{video_path.stem}_document.md"
    )

    try:
        if not document_path.exists():
            synthesize_video(str(video_path), str(document_path))

        report = ingest_document(str(document_path), str(video_path))
    except ValueError as exc:
        print(f"Skipped {video_path.name}: {exc}")
        return {"file_name": video_path.name, "skipped": True}

    print(f"Building graph for {video_path.name}...")

    graph = build_graph_for_file(report["file_id"])

    report["triples_loaded"] = graph["triples_loaded"]

    return report

def process_pdf(pdf_path: Path) -> dict:
    print(f"Processing: {pdf_path.name}")

    if pdf_path.stat().st_size == 0:
        print(f"Skipped {pdf_path.name}: empty file.")
        return {"file_name": pdf_path.name, "skipped": True}

    report = ingest_pdf(
        pdf_path.read_bytes(),
        pdf_path.name,
    )

    if report.chunks_created == 0:
        print(f"Skipped {pdf_path.name}: no readable text found.")
        return {"file_name": pdf_path.name, "skipped": True}

    print(f"Stored {report.chunks_created} chunks for {pdf_path.name}")
    print(f"Building graph for {pdf_path.name}...")

    graph = build_graph_for_file(report.file_id)

    return {
        "file_id": report.file_id,
        "file_name": report.file_name,
        "chunks_created": report.chunks_created,
        "triples_loaded": graph["triples_loaded"],
    }
    
    

def main(folder: str) -> int:
    folder_path = Path(folder)

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    extensions = {".pdf", ".mp4", ".mov", ".avi", ".mkv", ".webm"}

    files = sorted(
        path for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    if not files:
        print(f"No supported files found in {folder_path}")
        return 0

    print(f"Found {len(files)} files to process.")

    failed = 0
    skipped = 0

    for index, file_path in enumerate(files):
        try:
            if file_path.suffix.lower() == ".pdf":
                report = process_pdf(file_path)
            else:
                report = process_video(file_path)

            if report.get("skipped", False):
                skipped += 1
            else:
                print(f"Finished: {file_path.name}")

        except Exception as exc:
            failed += 1
            print(f"Failed {file_path.name}: {exc}")

        if index < len(files) - 1:
            time.sleep(5)

    completed = len(files) - skipped - failed
    print(f"Completed: {completed}; skipped: {skipped}; failed: {failed}")

    return 1 if failed else 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ingest_folder.py FOLDER_PATH")

    raise SystemExit(main(sys.argv[1]))