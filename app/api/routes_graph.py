from fastapi import APIRouter
from pydantic import BaseModel

from app.services.graph_service import build_graph_for_file
from app.vector_store import graph_store


router = APIRouter(prefix="/graph", tags=["graph"])


class GraphTotals(BaseModel):
    nodes: int
    relationships: int


class ExtractionSummary(BaseModel):
    batches_used: int
    raw_count: int
    deduped_count: int


class GraphBuildResponse(BaseModel):
    file_id: str
    file_name: str | None
    triples_loaded: int
    graph_totals: GraphTotals
    extraction: ExtractionSummary


@router.post("/build/{file_id}", response_model=GraphBuildResponse)
def build_graph_endpoint(file_id: str) -> GraphBuildResponse:
    result = build_graph_for_file(file_id)
    return GraphBuildResponse(**result)


@router.get("/stats", response_model=GraphTotals)
def graph_stats_endpoint() -> GraphTotals:
    return GraphTotals(**graph_store.count_graph())
