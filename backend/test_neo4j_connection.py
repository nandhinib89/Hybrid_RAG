import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def test_connection():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        driver.verify_connectivity()

        print("\n==============================")
        print("NEO4J CONNECTION TEST")
        print("==============================")
        print("✅ Successfully connected to Neo4j Aura")

    finally:
        driver.close()


if __name__ == "__main__":

    try:
        test_connection()

    except Exception as e:
        print("\n❌ NEO4J CONNECTION FAILED")
        print(f"Error: {e}")