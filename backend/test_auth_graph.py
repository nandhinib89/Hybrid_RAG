import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

query = """
MATCH (entity:Entity)
WHERE toLower(entity.name) CONTAINS "auth"
   OR toLower(entity.name) CONTAINS "session"
   OR toLower(entity.name) CONTAINS "credential"
   OR toLower(entity.name) CONTAINS "user"

OPTIONAL MATCH (entity)-[r]->(related:Entity)

RETURN
    entity.name AS entity,
    entity.type AS entity_type,
    type(r) AS relationship,
    related.name AS related_entity,
    related.type AS related_type

ORDER BY entity, related_entity
"""

try:
    with driver.session() as session:
        result = session.run(query)

        for record in result:
            print(
                f"{record['entity']} "
                f"({record['entity_type']}) "
                f"--[{record['relationship']}]--> "
                f"{record['related_entity']} "
                f"({record['related_type']})"
            )

finally:
    driver.close()