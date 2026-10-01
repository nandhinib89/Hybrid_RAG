import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import (
    LLMGraphTransformer
)
from langchain_core.documents import Document

try:
    # When imported by FastAPI
    from backend.ingestion import process_file
except ModuleNotFoundError:
    # When running directly:
    # python backend/knowledge_graph.py
    from ingestion import process_file


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

NEO4J_URI = os.getenv(
    "NEO4J_URI"
)

NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME"
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD"
)


# --------------------------------------------------
# Neo4j connection
# --------------------------------------------------

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# --------------------------------------------------
# Graph Transformer
# --------------------------------------------------

graph_transformer = LLMGraphTransformer(
    llm=llm,

    # ------------------------------------------------
    # Allowed node types
    # ------------------------------------------------

    allowed_nodes=[
        "Application",
        "Service",
        "Database",
        "Cache",
        "Queue",
        "Storage",
        "Framework",
        "Library",
        "Language",
        "Infrastructure",
        "Event",
        "API",
        "Component"
    ],

    # ------------------------------------------------
    # Allowed relationship types
    # ------------------------------------------------

    allowed_relationships=[
        "USES",
        "DEPENDS_ON",
        "COMMUNICATES_WITH",
        "STORES_DATA_IN",
        "CACHES_DATA_IN",
        "PUBLISHES_TO",
        "CONSUMES_FROM",
        "IMPLEMENTS",
        "CONTAINS"
    ]
)


# --------------------------------------------------
# Clear existing graph
# --------------------------------------------------

def clear_graph():
    """
    Delete the existing Neo4j graph.

    The application currently supports one active
    document at a time, so this is called during
    document ingestion before storing the newly
    extracted graph.
    """

    with driver.session() as session:

        session.run(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )

    print(
        "🗑️ Existing graph cleared"
    )


# --------------------------------------------------
# Store Graph
# --------------------------------------------------

def store_graph(graph_documents):
    """
    Store extracted entities and relationships
    in Neo4j.

    MERGE is used so duplicate entities and
    relationships within the current document
    are not blindly created.
    """

    node_count = 0
    relationship_count = 0

    with driver.session() as session:

        for graph_doc in graph_documents:

            # --------------------------------------
            # Create entity nodes
            # --------------------------------------

            for node in graph_doc.nodes:

                query = """
                MERGE (n:Entity {id: $id})

                SET
                    n.name = $name,
                    n.type = $type
                """

                session.run(
                    query,
                    id=node.id,
                    name=node.id,
                    type=node.type
                )

                node_count += 1

            # --------------------------------------
            # Create relationships
            # --------------------------------------

            for relationship in graph_doc.relationships:

                source = relationship.source.id
                target = relationship.target.id

                # Convert relationship names into
                # valid Neo4j relationship types.
                relationship_type = (
                    relationship.type
                    .upper()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                query = f"""
                MATCH (
                    source:Entity
                    {{id: $source}}
                )

                MATCH (
                    target:Entity
                    {{id: $target}}
                )

                MERGE (
                    source
                )-[:{relationship_type}]->(
                    target
                )
                """

                session.run(
                    query,
                    source=source,
                    target=target
                )

                relationship_count += 1

    print(
        f"Nodes processed: {node_count}"
    )

    print(
        f"Relationships processed: "
        f"{relationship_count}"
    )

    return {
        "nodes": node_count,
        "relationships": relationship_count
    }


# --------------------------------------------------
# Add document to Knowledge Graph
# --------------------------------------------------

def add_document_to_knowledge_graph(
    file_path,
    processed_result=None
):
    """
    Process one document, extract entities and
    relationships, and create a fresh Neo4j
    knowledge graph for that document.

    The application currently supports one active
    document at a time, so each successful ingestion
    replaces the previous graph.

    If processed_result is provided, it is reused
    so the document is processed only once.
    """

    print(
        "\n=============================="
    )

    print(
        "KNOWLEDGE GRAPH INGESTION"
    )

    print(
        "=============================="
    )

    # --------------------------------------------------
    # Process document
    # --------------------------------------------------

    if processed_result is None:

        result = process_file(
            file_path
        )

    else:

        result = processed_result

    print(
        f"\nDocument: "
        f"{result['file_name']}"
    )

    print(
        f"Chunks: "
        f"{result['chunk_count']}"
    )

    # --------------------------------------------------
    # Convert chunks to LangChain Documents
    # --------------------------------------------------

    documents = []

    for chunk in result["chunks"]:

        documents.append(
            Document(
                page_content=chunk["text"],

                metadata={
                    "document_id": (
                        chunk["metadata"]
                        ["document_id"]
                    ),

                    "file_name": (
                        chunk["metadata"]
                        ["file_name"]
                    ),

                    "file_type": (
                        chunk["metadata"]
                        ["file_type"]
                    ),

                    "chunk_id": (
                        chunk["metadata"]
                        ["chunk_id"]
                    ),

                    "page_number": (
                        chunk["metadata"]
                        .get("page_number")
                    )
                }
            )
        )

    print(
        "Documents sent to graph extraction: "
        f"{len(documents)}"
    )

    # --------------------------------------------------
    # Extract graph
    # --------------------------------------------------

    print(
        "\nExtracting entities and relationships..."
    )

    graph_documents = (
        graph_transformer
        .convert_to_graph_documents(
            documents
        )
    )

    print(
        "Graph documents created: "
        f"{len(graph_documents)}"
    )

    # --------------------------------------------------
    # Clear previous graph
    #
    # This happens only after the new graph has been
    # extracted successfully. If extraction fails,
    # the previous graph remains available.
    # --------------------------------------------------

    clear_graph()

    # --------------------------------------------------
    # Store new graph
    # --------------------------------------------------

    graph_stats = store_graph(
        graph_documents
    )

    print(
        "\n✅ Knowledge graph ingestion completed"
    )

    return {
        "document_id": (
            result["document_id"]
        ),

        "file_name": (
            result["file_name"]
        ),

        "file_type": (
            result["file_type"]
        ),

        "chunk_count": (
            result["chunk_count"]
        ),

        "nodes_processed": (
            graph_stats["nodes"]
        ),

        "relationships_processed": (
            graph_stats["relationships"]
        )
    }


# --------------------------------------------------
# Direct execution test
# --------------------------------------------------

if __name__ == "__main__":

    FILE_PATH = (
        "data/uploads/"
        "spotify_web_app_architecture.pdf"
    )

    try:

        result = (
            add_document_to_knowledge_graph(
                FILE_PATH
            )
        )

        print(
            "\nResult:"
        )

        print(result)

    finally:

        driver.close()