from app.vector_store.qdrant_store import get_chunks_by_file_id
from collections import Counter, defaultdict
import re

import sys
from pathlib import Path

from app.vector_store.graph_store import get_all_triples
from app.vector_store.qdrant_store import list_documents

def _reconstruct_document(file_id: str) -> str:
    chunks = get_chunks_by_file_id(file_id)
    paragraphs = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _count_entities_by_file(triples: list[dict]) -> dict[str, Counter[str]]:
    counts_by_file = defaultdict(Counter)

    for triple in triples:
        file_id = triple["file_id"]
        if not file_id:
            continue

        for name in (triple["subject"], triple["object"]):
            if name:
                counts_by_file[file_id][name] += 1

    return dict(counts_by_file)


def _count_crossing_relationships(
    triples: list[dict],
    source_entities: set[str],
    target_entities: set[str],
) -> int:
    target_only = target_entities - source_entities
    count = 0

    for triple in triples:
        subject = triple["subject"]
        object_name = triple["object"]

        if (
            subject in source_entities and object_name in target_only
        ) or (
            object_name in source_entities and subject in target_only
        ):
            count += 1

    return count


def _related_file_ids(
    source_id: str,
    triples: list[dict],
    entities_by_file: dict[str, set[str]],
) -> list[str]:
    source_entities = entities_by_file[source_id]
    scores = {}

    for target_id, target_entities in entities_by_file.items():
        if target_id == source_id:
            continue

        shared = len(source_entities & target_entities)
        crossing = _count_crossing_relationships(
            triples, source_entities, target_entities
        )
        score = shared + crossing
        if score > 0:
            scores[target_id] = score

    ranked = sorted(scores, key=lambda file_id: scores[file_id], reverse=True)
    if not ranked:
        return []

    minimum_score = scores[ranked[0]] / 2
    return [file_id for file_id in ranked if scores[file_id] >= minimum_score][:3]


_INLINE_STOP_TERMS = {
    "system", "data", "date", "detail", "details",
    "description", "descriptions", "information",
    "list", "view", "screen", "button", "buttons",
    "field", "fields", "column", "columns",
    "row", "rows", "name", "names",
    "table layout", "grid layout", "user interface", "software interface",
}


def _inline_link_targets(
    source_id: str,
    counts_by_file: dict[str, Counter[str]],
) -> dict[str, str]:
    best: dict[str, tuple[str, int]] = {}

    for file_id, counts in counts_by_file.items():
        if file_id == source_id:
            continue

        for entity, count in counts.items():
            if count < 2 or len(entity) < 5 or entity.lower() in _INLINE_STOP_TERMS:
                continue

            previous = best.get(entity)
            if previous is None or count > previous[1]:
                best[entity] = (file_id, count)

    return {entity: file_id for entity, (file_id, _) in best.items()}


def _add_inline_links(
    document: str,
    targets: dict[str, str],
    note_names: dict[str, str],
) -> str:
    linked = 0

    for entity in sorted(targets, key=len, reverse=True):
        if linked >= 8:
            break

        pattern = re.compile(
            r"```[\s\S]*?```|~~~[\s\S]*?~~~|`+[^`\n]*`+"
            r"|\[\[[^\n]*?\]\]"
            r"|!?\[[^\]\n]*\]\([^\n]*?\)"
            rf"|(?<!\w)({re.escape(entity)})(?!\w)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(document):
            found = match.group(1)
            if found is None:
                continue

            target_id = targets[entity]
            link = f"[[{note_names[target_id]}|{found}]]"
            document = document[:match.start()] + link + document[match.end():]
            linked += 1
            break

    return document


def _build_note_names(files: list[dict]) -> dict[str, str]:
    note_names = {}
    used_names = set()

    for doc in sorted(files, key=lambda doc: doc["file_id"]):
        title = Path(doc["file_name"] or doc["file_id"]).stem
        title = title.replace("_", " ").replace("-", " ").title()

        base = re.sub(r'[\x00-\x1f<>:"/\\|?*\[\]#^]', "", title).strip(" .")
        base = base.encode("utf-8")[:180].decode("utf-8", errors="ignore")
        base = base or "Untitled"

        name = base
        number = 2

        while name.casefold() in used_names:
            name = f"{base} ({number})"
            number += 1

        note_names[doc["file_id"]] = name
        used_names.add(name.casefold())

    return note_names


def _strip_leading_title(document: str) -> str:
    lines = document.splitlines()

    while lines and (
        not lines[0].strip()
        or lines[0].lstrip().startswith("# ")
    ):
        lines.pop(0)

    return "\n".join(lines).strip()


def export_to_obsidian(
    output_dir: str,
    source_type: str | None = None,
) -> dict:
    files = [
        doc for doc in list_documents()
        if doc["source_type"] in {"video", "pdf"}
        and (source_type is None or doc["source_type"] == source_type)
    ]

    note_names = _build_note_names(files)
    documents = {
        doc["file_id"]: _reconstruct_document(doc["file_id"])
        for doc in files
    }

    triples = get_all_triples()
    all_counts = _count_entities_by_file(triples)

    counts = {
        file_id: all_counts.get(file_id, Counter())
        for file_id in note_names
    }

    entities = {
        file_id: set(counter)
        for file_id, counter in counts.items()
    }

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    for doc in files:
        file_id = doc["file_id"]
        document = documents[file_id]

        if doc["source_type"] == "video":
            document = _strip_leading_title(document)

        document = document or "_No saved text was available for this file._"

        targets = _inline_link_targets(file_id, counts)
        document = _add_inline_links(document, targets, note_names)
        related = _related_file_ids(file_id, triples, entities)

        lines = [f"# {note_names[file_id]}", "", document, ""]

        if related:
            lines.append("## See also")

            for target_id in related:
                lines.append(f"- [[{note_names[target_id]}]]")

            lines.append("")

        path = output / f"{note_names[file_id]}.md"
        path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "notes_written": len(files),
        "path": str(output.resolve()),
    }


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) > 1 else "obsidian_vault"
    result = export_to_obsidian(destination)

    print(f"Wrote {result['notes_written']} notes to {result['path']}")