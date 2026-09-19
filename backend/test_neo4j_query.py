import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def test_graph_query():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        with driver.session() as session:

            result = session.run(
                """
                MATCH (source:Entity)-[r]->(target:Entity)
                RETURN
                    source.name AS source,
                    source.type AS source_type,
                    type(r) AS relationship,
                    r.type AS relationship_type,
                    target.name AS target,
                    target.type AS target_type
                LIMIT 20
                """
            )

            print("\n==============================")
            print("NEO4J GRAPH QUERY")
            print("==============================")

            for record in result:

                print(
                    f"\n{record['source']} "
                    f"({record['source_type']})"
                )

                print(
                    f"  --[{record['relationship']}"
                    f": {record['relationship_type']}]--> "
                )

                print(
                    f"{record['target']} "
                    f"({record['target_type']})"
                )

    finally:
        driver.close()


if __name__ == "__main__":

    try:
        test_graph_query()

    except Exception as e:
        print("\n❌ GRAPH QUERY FAILED")
        print(f"Error: {e}")