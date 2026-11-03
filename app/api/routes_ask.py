from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.answer_service import answer_question


router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural-language question.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="How many chunks to retrieve as context.",
    )
    file_id: str | None = Field(
        None,
        description="Restrict to a specific file_id.",
    )
    source_type: str | None = Field(
        None,
        description="Restrict to 'pdf' or 'video'.",
    )


class SourceModel(BaseModel):
    file_id: str | None
    file_name: str | None
    source_type: str | None
    page_number: int | None
    timestamp_seconds: int | None
    frame_number: int | None
    chunk_index: int | None
    score: float
    label: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]


@router.post("", response_model=AskResponse)
async def ask_endpoint(payload: AskRequest) -> AskResponse:
    result = answer_question(
        question=payload.question,
        top_k=payload.top_k,
        file_id=payload.file_id,
        source_type=payload.source_type,
    )
    return AskResponse(**asdict(result))
