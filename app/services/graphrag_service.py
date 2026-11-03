import json
import logging

from app.config import settings
from openai import BadRequestError, OpenAI

from app.embeddings.embedding_client import embed_text
from app.vector_store import qdrant_store

logger = logging.getLogger(__name__)

_ENTITY_PROMPT = """Prepare English entity names for knowledge-graph search.
Complete these two tasks in order, in ONE response:

1. TRANSLATE THE QUESTION.
Translate the entire question into natural English without answering it.
Preserve all named things, details, conditions, numbers and identifiers.
If the question is already English, keep it in English.
Put this complete English question in english_question.

2. EXTRACT FROM THE ENGLISH QUESTION.
Use the English question from step 1 to identify its named subjects and complete business-process or topic phrases.
Include named people, organisations, systems, products, documents, places, projects, activities and concepts when mentioned.
Keep each meaningful phrase together instead of splitting it into isolated words.
Use English Title Case for ordinary topic/process names. Preserve actual proper names, acronyms and identifiers.
Do not invent the unknown person, project, document, department or relationship that the question asks you to find.
Do not add generic question words or a vague standalone label for the requested answer.
Remove duplicates. For a greeting with no named thing or specific topic, entities is an empty list.

Return ONLY JSON with english_question first and entities second.
Do not answer the question or explain your work.
"""

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "english_question": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["english_question", "entities"],
    "additionalProperties": False,
}


def _extract_question_entities(question: str) -> list[str]:
    if not question.strip():
        return []

    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not configured for question entities.")
        return []

    try:
        with OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.EXTRACTION_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            # Retry one provider-side JSON-generation failure, never indefinitely.
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=settings.QUESTION_ENTITY_MODEL,
                        messages=[
                            {"role": "system", "content": _ENTITY_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        reasoning_effort="none",
                        temperature=0,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "question_entities",
                                "strict": True,
                                "schema": _ENTITY_SCHEMA,
                            },
                        },
                    )
                    break
                except BadRequestError as exc:
                    if attempt or exc.code != "json_validate_failed":
                        raise
                    logger.warning("Groq returned invalid JSON; retrying once.")

        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise RuntimeError("Entity extraction did not finish normally.")

        data = json.loads(choice.message.content or "")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        english_question = data.get("english_question")
        if not isinstance(english_question, str) or not english_question.strip():
            raise ValueError("The English question is missing.")
        logger.debug("English question: %s", english_question)
        entities = data.get("entities", [])

        if not isinstance(entities, list) or any(
            not isinstance(name, str) for name in entities
        ):
            raise ValueError("Expected a list of entity names.")

        return [name.strip() for name in entities if name.strip()]

    except Exception as exc:
        logger.warning("Question entity extraction failed: %s", exc)
        return []


def _vector_context(question: str, top_k: int) -> list[dict]:
    question_vector = embed_text(question)
    return qdrant_store.search(question_vector, top_k=top_k)