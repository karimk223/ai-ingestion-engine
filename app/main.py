from fastapi import FastAPI
from app.api.routes_ask import router as ask_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_search import router as search_router
from app.api.routes_document import router as document_router
from app.api.routes_graphrag import router as graphrag_router
from app.api.routes_extract import router as extract_router
from app.api.routes_graph import router as graph_router
from app.api.routes_minutes import router as minutes_router
from app.api.routes_entity_resolution import router as entity_resolution_router
from app.api.routes_communities import router as communities_router

app = FastAPI(
    title="AI Ingestion Engine",
    description="Multimodal RAG pipeline for PDFs and videos.",
    version="0.3.0",
)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


app.include_router(ask_router)
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(document_router)
app.include_router(graphrag_router)
app.include_router(extract_router)
app.include_router(graph_router)
app.include_router(minutes_router)
app.include_router(entity_resolution_router)
app.include_router(communities_router)