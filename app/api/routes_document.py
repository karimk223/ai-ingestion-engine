from pydantic import BaseModel
from fastapi import APIRouter

from app.vector_store.qdrant_store import list_documents


router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentInfo(BaseModel):
    file_id: str
    file_name: str | None
    source_type: str | None
    chunk_count: int


class DocumentsResponse(BaseModel):
    count: int
    documents: list[DocumentInfo]


@router.get("", response_model=DocumentsResponse)
async def list_documents_endpoint(
    name: str | None = None,
) -> DocumentsResponse:
    docs = list_documents()

    if name:
        needle = name.lower()
        docs = [
            d for d in docs
            if d.get("file_name") and needle in d["file_name"].lower()
        ]

    return DocumentsResponse(
        count=len(docs),
        documents=[DocumentInfo(**d) for d in docs],
    )
