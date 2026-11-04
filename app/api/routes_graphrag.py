from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.services.graphrag_service import answer_question


router = APIRouter(prefix="/graphrag", tags=["graphrag"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, pattern=r"\S")
    top_k: int = Field(5, ge=1, le=20)
    hops: int = Field(2, ge=1, le=4)
    use_graph: bool = True
    file_id: str | None = None


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    file_id: str | None = None
    file_name: str | None = None


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    vector_chunks_used: int
    graph_entities_resolved: list[str]
    graph_triples_used: list[Triple]


@router.post("/ask", response_model=AskResponse)
def graphrag_ask_endpoint(payload: AskRequest) -> AskResponse:
    result = answer_question(
        question=payload.question,
        top_k=payload.top_k,
        hops=payload.hops,
        use_graph=payload.use_graph,
        file_id=payload.file_id,
    )
    return AskResponse(**result)


@router.post("/ask-simple", response_model=AskResponse)
def graphrag_ask_simple_endpoint(
    question: str = Query(..., min_length=1, pattern=r"\S"),
    top_k: int = Query(5, ge=1, le=20),
    hops: int = Query(2, ge=1, le=4),
    use_graph: bool = True,
    file_id: str | None = None,
) -> AskResponse:
    result = answer_question(
        question=question,
        top_k=top_k,
        hops=hops,
        use_graph=use_graph,
        file_id=file_id,
    )
    return AskResponse(**result)