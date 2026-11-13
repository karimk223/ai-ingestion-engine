from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.minutes_service import generate_minutes
from app.vector_store.qdrant_store import list_documents


router = APIRouter(prefix="/minutes", tags=["minutes"])


class MinutesResponse(BaseModel):
    file_id: str
    file_name: str | None
    minutes: str
    batches_used: int
    method: str


def _minutes_response(file_id: str) -> MinutesResponse:
    try:
        result = generate_minutes(file_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return MinutesResponse(**result)


@router.post("/by-name", response_model=MinutesResponse)
async def minutes_by_name_endpoint(name: str) -> MinutesResponse:
    docs = list_documents()
    needle = (name or "").strip().lower()

    if not needle:
        raise HTTPException(status_code=400, detail="Provide a name to search for.")

    matches = [
        doc for doc in docs
        if doc.get("file_name") and needle in doc["file_name"].lower()
    ]

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No ingested file matches the name '{name}'.",
        )

    videos = [doc for doc in matches if doc.get("source_type") == "video"]

    if not videos:
        names = ", ".join(sorted({doc["file_name"] for doc in matches}))
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{name}' matches {names}, which is not a video. "
                "Minutes of Meeting are only generated for videos."
            ),
        )

    distinct_ids = {video["file_id"] for video in videos}
    if len(distinct_ids) > 1:
        listing = "; ".join(
            f"{video['file_name']} (id {video['file_id'][:8]}…)"
            for video in videos
        )
        raise HTTPException(
            status_code=400,
            detail=f"'{name}' matches more than one video: {listing}. Please be more specific.",
        )

    return _minutes_response(videos[0]["file_id"])


@router.post("/{file_id}", response_model=MinutesResponse)
async def minutes_endpoint(file_id: str) -> MinutesResponse:
    return _minutes_response(file_id)