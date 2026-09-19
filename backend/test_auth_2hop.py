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
WHERE toLower(entity.name) = "auth"

MATCH path = (entity)-[*1..2]-(related:Entity)

RETURN
    [node IN nodes(path) | node.name] AS nodes,
    [rel IN relationships(path) | type(rel)] AS relationships

LIMIT 50
"""


try:
    with driver.session() as session:
        results = session.run(query)

        print("\n==============================")
        print("AUTH 2-HOP GRAPH TEST")
        print("==============================")

        found = False

        for record in results:
            found = True

            nodes = record["nodes"]
            relationships = record["relationships"]

            print("\nPath:")

            for i, node in enumerate(nodes):
                print(f"  {node}")

                if i < len(relationships):
                    print(f"    --[{relationships[i]}]-->")

        if not found:
            print("\n❌ No 1–2 hop Entity paths found.")

finally:
    driver.close()