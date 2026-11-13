from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.entity_resolution_service import resolve_entities


router = APIRouter(prefix="/graph", tags=["graph"])


class GraphTotals(BaseModel):
    nodes: int
    relationships: int


class ResolveResponse(BaseModel):
    applied: bool
    entity_count_before: int
    merges: list[dict]
    graph_totals: GraphTotals


@router.post("/resolve-entities", response_model=ResolveResponse)
def resolve_entities_endpoint(apply: bool = False) -> ResolveResponse:
    try:
        result = resolve_entities(apply=apply)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ResolveResponse(**result)