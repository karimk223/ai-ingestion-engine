import json

from openai import OpenAI

from app.config import settings
from app.vector_store.graph_store import count_graph, list_entity_names, merge_entities


_RESOLUTION_PROMPT = (
    "You are reviewing entity names from an ERP knowledge graph. "
    "Propose a merge only when two names refer to the same real thing. "
    "Similar names can refer to different things. For example, "
    "Size Order and Sales Order are distinct. When in doubt, skip the pair. "
    'Return JSON in this format: {"merges": [{"from": "duplicate name", '
    '"into": "name to keep", "reason": "why these are the same"}]}. '
    'If there are no duplicates, return {"merges": []}.'
)


def propose_merges() -> list[dict]:
    names = list_entity_names()
    if len(names) < 2:
        return []

    client = OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=settings.QUESTION_ENTITY_MODEL,
        messages=[
            {"role": "system", "content": _RESOLUTION_PROMPT},
            {"role": "user", "content": "Entity names:\n" + "\n".join(names)},
        ],
        response_format={"type": "json_object"},
        timeout=settings.EXTRACTION_TIMEOUT_SECONDS,
        extra_body={"reasoning_effort": "none"},
    )

    raw = response.choices[0].message.content if response.choices else ""
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return []

    proposals = data.get("merges") if isinstance(data, dict) else None
    if not isinstance(proposals, list):
        return []

    known_names = set(names)
    valid = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue

        duplicate = str(proposal.get("from", "")).strip()
        keep = str(proposal.get("into", "")).strip()
        if duplicate != keep and duplicate in known_names and keep in known_names:
            valid.append({
                "from": duplicate,
                "into": keep,
                "reason": str(proposal.get("reason", "")).strip(),
            })

    return valid

def resolve_entities(apply: bool = False) -> dict:
    entity_count_before = len(list_entity_names())
    merges = propose_merges()

    applied = apply and bool(merges)
    if applied:
        for proposal in merges:
            merge_entities(proposal["from"], proposal["into"])

    return {
        "applied": applied,
        "entity_count_before": entity_count_before,
        "merges": merges,
        "graph_totals": count_graph(),
    }