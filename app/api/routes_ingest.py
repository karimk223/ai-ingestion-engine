from dataclasses import asdict

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import settings
from app.services.ingestion_service import ingest_pdf


router = APIRouter(prefix="/ingest", tags=["ingest"])


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
