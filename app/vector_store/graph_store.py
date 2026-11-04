from neo4j import GraphDatabase

from app.config import settings


_driver = None


def get_driver():
    """Return the shared Neo4j connection manager."""
    global _driver

    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

    return _driver


def verify_connectivity() -> None:
    """Check that Neo4j is reachable and the login works."""
    get_driver().verify_connectivity()


def load_triples(
    file_id: str,
    file_name: str | None,
    triples: list[dict],
) -> int:
    if not triples:
        return 0

    cypher = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $object})
    MERGE (s)-[r:REL {predicate: $predicate, file_id: $file_id}]->(o)
    SET r.file_name = $file_name
    """

    driver = get_driver()
    loaded = 0

    with driver.session() as session:
        for t in triples:
            session.run(
                cypher,
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                file_id=file_id,
                file_name=file_name,
            )
            loaded += 1

    return loaded


def count_graph() -> dict:
    driver = get_driver()

    with driver.session() as session:
        nodes = session.run(
            "MATCH (n) RETURN count(n) AS c"
        ).single()["c"]

        rels = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]

    return {"nodes": nodes, "relationships": rels}


def clear_file(file_id: str) -> None:
    driver = get_driver()

    with driver.session() as session:
        session.run(
            "MATCH ()-[r:REL {file_id: $fid}]->() DELETE r",
            fid=file_id,
        )

        session.run(
            "MATCH (n:Entity) WHERE NOT (n)--() DELETE n"
        )


def find_entities(names: list[str]) -> list[str]:
    if not names:
        return []

    cypher = """
    UNWIND $names AS candidate
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower(candidate)
       OR toLower(candidate) CONTAINS toLower(e.name)
    RETURN DISTINCT e.name AS name
    """

    driver = get_driver()

    with driver.session() as session:
        rows = session.run(cypher, names=names)
        return [r["name"] for r in rows]


def neighborhood_triples(
    entity_names: list[str],
    hops: int = 2,
    limit: int = 60,
    file_id: str | None = None,
) -> list[dict]:
    if not entity_names:
        return []

    hops = max(1, min(int(hops), 4))

    cypher = f"""
    MATCH (a:Entity)
    WHERE a.name IN $names
    MATCH path = (a)-[:REL*1..{hops}]-(b:Entity)
    WHERE $file_id IS NULL
       OR all(rel IN relationships(path) WHERE rel.file_id = $file_id)
    WITH relationships(path) AS rels
    UNWIND rels AS r
    WITH startNode(r) AS s, r AS r, endNode(r) AS o
    RETURN DISTINCT s.name AS subject,
                    r.predicate AS predicate,
                    o.name AS object,
                    r.file_id AS file_id,
                    r.file_name AS file_name
    LIMIT $limit
    """

    driver = get_driver()

    with driver.session() as session:
        rows = session.run(
            cypher,
            names=entity_names,
            limit=limit,
            file_id=file_id,
        )

        return [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "file_id": r["file_id"],
                "file_name": r["file_name"],
            }
            for r in rows
        ]
