from fastapi import APIRouter
from pydantic import BaseModel

from app.services.extraction_service import extract_from_file


router = APIRouter(prefix="/extract", tags=["extract"])


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractResponse(BaseModel):
    file_id: str
    file_name: str | None
    triples: list[Triple]
    batches_used: int
    raw_count: int
    deduped_count: int


@router.post("/{file_id}", response_model=ExtractResponse)
def extract_endpoint(file_id: str) -> ExtractResponse:
    result = extract_from_file(file_id)
    return ExtractResponse(**result)
