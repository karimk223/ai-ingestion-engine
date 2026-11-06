from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import PROJECT_ROOT, settings
from app.services.ingestion_service import ingest_pdf, ingest_video


router = APIRouter(prefix="/ingest", tags=["ingest"])

MAX_VIDEO_MB = 200
ALLOWED_VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@router.post("/pdf", status_code=status.HTTP_201_CREATED)
async def ingest_pdf_endpoint(file: UploadFile = File(...)):

    # Check the reported file type.
    if file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Expected a PDF file, "
                f"got content_type={file.content_type!r}"
            ),
        )

    # Check that the filename ends with .pdf.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a .pdf extension.",
        )

    # Read the actual contents of the uploaded file.
    pdf_bytes = await file.read()

    # Convert the configured size limit into bytes.
    max_bytes = settings.MAX_PDF_MB * 1024 * 1024

    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"PDF is {len(pdf_bytes) // (1024 * 1024)} MB; "
                f"limit is {settings.MAX_PDF_MB} MB."
            ),
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Run our existing PDF processing function.
    report = ingest_pdf(
        pdf_bytes,
        file_name=file.filename,
    )

    return asdict(report)

@router.post("/video", status_code=status.HTTP_201_CREATED)
async def ingest_video_endpoint(file: UploadFile = File(...)):
    # Check that the uploaded file has a supported video extension.
    if not file.filename or not file.filename.lower().endswith(ALLOWED_VIDEO_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a video: {', '.join(ALLOWED_VIDEO_EXT)}.",
        )

    max_bytes = MAX_VIDEO_MB * 1024 * 1024
    video_bytes = await file.read(max_bytes + 1)

    if len(video_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Video limit is {MAX_VIDEO_MB} MB.",
        )

    if not video_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Save a temporary copy because our video functions need a file path.
    uploads_dir = PROJECT_ROOT / "storage" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()

    with NamedTemporaryFile(
        dir=uploads_dir,
        suffix=suffix,
        delete=False,
    ) as temp:
        video_path = Path(temp.name)

    try:
        video_path.write_bytes(video_bytes)

        report = await run_in_threadpool(
            ingest_video,
            str(video_path),
            file_name=Path(file.filename).name,
        )

        return asdict(report)

    finally:
        video_path.unlink(missing_ok=True)
