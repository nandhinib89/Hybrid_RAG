import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def extract_entities(question):
    """
    Extract concise architectural concepts that can be
    used to search the knowledge graph.
    """

    prompt = f"""
You are helping retrieve information from a knowledge graph
about a software architecture.

Extract the important architectural concepts from the user's question.

Return SHORT search terms, not long natural-language phrases.

Prefer important nouns or technology/service concepts that
are likely to appear in entity names.

Examples:

Question:
"What databases are used for authentication and session management?"

Return:
authentication, session

Question:
"Which technology handles full-text search?"

Return:
search

Question:
"What does the catalog service store?"

Return:
catalog

Question:
"Which services use PostgreSQL?"

Return:
PostgreSQL

Do NOT return generic words such as:
what, which, used, things, service, databases, management

Return ONLY a comma-separated list.

Question:
{question}
"""

    response = llm.invoke(prompt)

    entities = [
        item.strip()
        for item in response.content.split(",")
        if item.strip()
    ]

    return entities


# -----------------------------
# Neo4j Retrieval
# -----------------------------

def search_graph(question, entities):
    """
    Retrieve candidate entities from Neo4j, then use the LLM
    to identify which candidates are relevant to the question.
    """

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    candidate_query = """
    UNWIND $entities AS search_term

    MATCH (entity:Entity)
    WHERE
        toLower(entity.name) CONTAINS toLower(search_term)
        OR toLower(search_term) CONTAINS toLower(entity.name)

    WITH DISTINCT entity

    RETURN
        entity.name AS entity,
        entity.type AS entity_type

    ORDER BY entity
    LIMIT 20
    """

    try:
        with driver.session() as session:
            result = session.run(
                candidate_query,
                entities=entities
            )

            candidates = [record.data() for record in result]

    finally:
        driver.close()

    if not candidates:
        return []

    # ---------------------------------------------------------
    # Ask the LLM which candidate entities are relevant
    # ---------------------------------------------------------

    candidate_text = "\n".join(
        f"- {item['entity']} ({item['entity_type']})"
        for item in candidates
    )

    prompt = f"""
You are resolving entities for a software architecture
knowledge graph.

User question:
{question}

Candidate entities found in the knowledge graph:

{candidate_text}

Identify the candidate entities that are directly relevant
to answering the user's question.

Do not invent entities.
Only select entities from the candidate list.

Return ONLY the exact entity names, one per line.

If a service is responsible for the concept in the question,
prefer that service over an unrelated service that happens
to contain a similar word.

For example, if the question is about authentication and
session management, prefer an authentication/identity
service over a playback session service.
"""

    response = llm.invoke(prompt)

    selected_names = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    # Keep only names that actually came from Neo4j
    candidate_names = {
        item["entity"]
        for item in candidates
    }

    selected_names = [
        name
        for name in selected_names
        if name in candidate_names
    ]

    if not selected_names:
        return []

    # ---------------------------------------------------------
    # Retrieve relationships for selected entities
    # ---------------------------------------------------------

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    relationship_query = """
    MATCH (entity:Entity)-[r]->(related:Entity)
    WHERE entity.name IN $selected_entities

    RETURN
        entity.name AS entity,
        entity.type AS entity_type,
        type(r) AS relationship,
        related.name AS related_entity,
        related.type AS related_type

    ORDER BY entity, related_entity
    LIMIT 30
    """

    try:
        with driver.session() as session:
            result = session.run(
                relationship_query,
                selected_entities=selected_names
            )

            return [record.data() for record in result]

    finally:
        driver.close()

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    question = "What databases are used for authentication and session management?"

    print("\n==============================")
    print("GRAPH RETRIEVAL TEST")
    print("==============================")

    print(f"\nQuestion:")
    print(question)

    # Step 1
    entities = extract_entities(question)

    print("\nExtracted entities/concepts:")
    for entity in entities:
        print(f"  - {entity}")

    # Step 2
    results = search_graph(question, entities)

    print("\nGraph results:")

    if not results:
        print("  No matching graph entities found.")

    else:
        for result in results:
            print(
                f"  {result['entity']} "
                f"--[{result['relationship']}]--> "
                f"{result['related_entity']}"
            )