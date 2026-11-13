from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.community_service import build_community_summaries


router = APIRouter(prefix="/graph", tags=["graph"])


class CommunitySummary(BaseModel):
    community: int
    entities: list[str]
    size: int
    summary: str


class CommunitiesResponse(BaseModel):
    community_count: int
    summaries: list[CommunitySummary]


@router.post("/communities", response_model=CommunitiesResponse)
def communities_endpoint(min_size: int = 2) -> CommunitiesResponse:
    try:
        result = build_community_summaries(min_size=min_size)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CommunitiesResponse(**result)