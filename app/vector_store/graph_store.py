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


def list_entity_names() -> list[str]:
    """Return every entity name currently in the graph."""
    driver = get_driver()
    with driver.session() as session:
        rows = session.run("MATCH (e:Entity) RETURN e.name AS name ORDER BY e.name")
        return [r["name"] for r in rows]


def merge_entities(from_name: str, into_name: str) -> None:
    """Move a duplicate's relationships to the entity we want to keep."""
    if from_name == into_name:
        return

    params = {"from": from_name, "into": into_name}

    with get_driver().session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                """
                MATCH (f:Entity {name: $from})-[r:REL]->(x:Entity)
                MATCH (t:Entity {name: $into})
                WHERE x <> t
                MERGE (t)-[nr:REL {predicate: r.predicate, file_id: r.file_id}]->(x)
                SET nr.file_name = r.file_name
                """,
                params,
            ).consume()

            tx.run(
                """
                MATCH (y:Entity)-[r:REL]->(f:Entity {name: $from})
                MATCH (t:Entity {name: $into})
                WHERE y <> t
                MERGE (y)-[nr:REL {predicate: r.predicate, file_id: r.file_id}]->(t)
                SET nr.file_name = r.file_name
                """,
                params,
            ).consume()

            tx.run(
                """
                MATCH (f:Entity {name: $from}), (:Entity {name: $into})
                DETACH DELETE f
                """,
                params,
            ).consume()

            tx.commit()

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


def detect_communities() -> list[dict]:
    """Group related entities and return the relationships inside each group."""
    graph_name = "ingestion_engine_communities"

    with get_driver().session() as session:
        session.run(
            """
            CALL gds.graph.project(
                $name,
                'Entity',
                {REL: {orientation: 'UNDIRECTED'}}
            )
            """,
            name=graph_name,
        ).consume()

        try:
            session.run(
                "CALL gds.louvain.write($name, {writeProperty: 'community'})",
                name=graph_name,
            ).consume()

            communities = session.run("""
                MATCH (e:Entity)
                WHERE e.community IS NOT NULL
                RETURN e.community AS community, collect(e.name) AS entities
                ORDER BY size(entities) DESC
            """).data()

            for group in communities:
                group["triples"] = session.run(
                    """
                    MATCH (s:Entity)-[r:REL]->(o:Entity)
                    WHERE s.name IN $names AND o.name IN $names
                    RETURN DISTINCT s.name AS subject,
                                    r.predicate AS predicate,
                                    o.name AS object
                    """,
                    names=group["entities"],
                ).data()

            return communities
        finally:
            session.run(
                "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                name=graph_name,
            ).consume()