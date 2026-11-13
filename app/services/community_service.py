import logging

from app.config import settings
from app.embeddings.embedding_client import embed_text
from app.llm.groq_client import generate_answer
from app.vector_store.graph_store import detect_communities
from app.vector_store.qdrant_store import search


logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = (
    "Summarize one knowledge graph community in a paragraph of 3–5 sentences. "
    "Describe the main entities, their relationships, and any process they form. "
    "Use only the supplied relationships and source passages. "
    "If the evidence is too thin, say so and describe which entities are grouped."
)


def _grounding_passages(entities: list[str], top_k: int = 4) -> str:
    if not entities:
        return ""

    query = ", ".join(entities)
    try:
        hits = search(embed_text(query), top_k=top_k)
    except Exception as exc:
        logger.warning("Community source lookup failed: %s", exc)
        return ""

    passages = [hit["text"].strip() for hit in hits if hit.get("text")]
    return "\n".join(f"- {passage}" for passage in passages if passage)


def _summarize_community(entities: list[str], triples: list[dict]) -> str:
    relationships = "\n".join(
        f"({t['subject']}) -{t['predicate']}-> ({t['object']})"
        for t in triples
    )
    passages = _grounding_passages(entities)

    return generate_answer(
        system_prompt=_SUMMARY_PROMPT,
        user_prompt=(
            f"ENTITIES:\n{', '.join(entities)}\n\n"
            f"RELATIONSHIPS:\n{relationships or '(none)'}\n\n"
            f"SOURCE PASSAGES:\n{passages or '(none retrieved)'}"
        ),
    ).strip()


def build_community_summaries(min_size: int = 2) -> dict:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required to summarize communities.")

    summaries = []
    for group in detect_communities():
        entities = group["entities"]
        if len(entities) < min_size:
            continue

        summaries.append({
            "community": group["community"],
            "entities": entities,
            "size": len(entities),
            "summary": _summarize_community(entities, group["triples"]),
        })

    return {"community_count": len(summaries), "summaries": summaries}