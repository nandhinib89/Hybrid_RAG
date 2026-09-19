import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase
from backend.llm import generate_answer

load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VECTOR_DB_PATH = "data/vector_db"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ---------------------------------------------------------
# Vector Retrieval
# ---------------------------------------------------------

def vector_search(question, k=5):
    """
    Retrieve relevant chunks from the FAISS vector database.
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    documents = vector_store.similarity_search(
        question,
        k=k
    )

    results = []

    for document in documents:
        results.append({
            "text": document.page_content,
            "metadata": document.metadata
        })

    return results


# ---------------------------------------------------------
# Knowledge Graph Retrieval
# ---------------------------------------------------------

def graph_search(question):
    """
    Retrieve graph context related to the question.

    This is intentionally kept simple for now.
    The goal is to integrate KG retrieval into the
    Hybrid RAG pipeline rather than perfect KG reasoning.
    """

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    query = """
    MATCH (entity:Entity)-[r]->(related:Entity)

    RETURN
        entity.name AS entity,
        entity.type AS entity_type,
        type(r) AS relationship,
        related.name AS related_entity,
        related.type AS related_type

    LIMIT 50
    """

    try:
        with driver.session() as session:
            result = session.run(query)

            return [record.data() for record in result]

    finally:
        driver.close()


# ---------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------

def hybrid_search(question, vector_k=5):
    """
    Retrieve information from both FAISS and Neo4j.
    """

    vector_results = vector_search(
        question,
        k=vector_k
    )

    graph_results = graph_search(question)

    return {
        "question": question,
        "vector_results": vector_results,
        "graph_results": graph_results
    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = "What databases are used for authentication and session management?"

    print("\n==============================")
    print("HYBRID RAG TEST")
    print("==============================")

    print("\nQuestion:")
    print(question)

    results = hybrid_search(question)

    print("\nRetrieval completed.")

    answer = generate_answer(
        question,
        results["vector_results"],
        results["graph_results"]
    )

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(answer)