"""Build a file's Neo4j graph from its extracted connections."""
import logging

from app.services.extraction_service import extract_from_file
from app.vector_store import graph_store


logger = logging.getLogger(__name__)


def build_graph_for_file(
    file_id: str,
    extraction: dict | None = None,
    *,
    reviewed_triples: list[dict] | None = None,
) -> dict:
    """Store extracted connections, optionally selecting a reviewed subset."""
    if extraction is None:
        extraction = extract_from_file(file_id)

    if extraction.get("file_id") != file_id:
        raise RuntimeError("Extraction report belongs to a different file.")

    candidates = extraction.get("triples")
    if not isinstance(candidates, list):
        raise RuntimeError("Extraction report must contain a list of connections.")

    triples = candidates if reviewed_triples is None else reviewed_triples
    if not isinstance(triples, list) or not triples:
        raise RuntimeError("No connections to save; the existing graph was left unchanged.")

    for triple in triples:
        if not isinstance(triple, dict) or not all(
            isinstance(triple.get(key), str) and triple[key].strip()
            for key in ("subject", "predicate", "object")
        ):
            raise RuntimeError("Invalid connection; the existing graph was left unchanged.")

    if reviewed_triples is not None and any(triple not in candidates for triple in triples):
        raise RuntimeError("Reviewed connections must come from this extraction report.")

    summary = {
        "batches_used": extraction["batches_used"],
        "raw_count": extraction["raw_count"],
        "deduped_count": extraction["deduped_count"],
    }
    file_name = extraction.get("file_name")

    graph_store.clear_file(file_id)
    loaded = graph_store.load_triples(file_id, file_name, triples)
    totals = graph_store.count_graph()

    logger.info(
        "Graph build for '%s': %d connections loaded (%d nodes, %d relationships total)",
        file_name,
        loaded,
        totals["nodes"],
        totals["relationships"],
    )

    report = {
        "file_id": file_id,
        "file_name": file_name,
        "triples_loaded": loaded,
        "graph_totals": totals,
        "extraction": summary,
    }
    if reviewed_triples is not None:
        report["review"] = {
            "candidates": len(candidates),
            "accepted": len(triples),
            "excluded": len(candidates) - len(triples),
        }
    return report
