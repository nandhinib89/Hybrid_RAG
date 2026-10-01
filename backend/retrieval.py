import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langsmith import traceable

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
# Cross-Encoder Model
# ---------------------------------------------------------

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------------------------------------------------
# Vector Retrieval
# ---------------------------------------------------------

@traceable(name="FAISS Vector Search")
def vector_search(question, k=10):
    """
    Retrieve candidate chunks from the FAISS vector database.

    FAISS performs semantic retrieval.
    A wider candidate set is retrieved so BM25 can rerank it.
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    documents_with_scores = vector_store.similarity_search_with_score(
      question,
      k=k
)

    results = []

    for document, score in documents_with_scores:
     results.append({
        "text": document.page_content,
        "metadata": document.metadata,
        "faiss_score": float(score)
    })

    return results


# ---------------------------------------------------------
# BM25 Reranking
# ---------------------------------------------------------

@traceable(name="BM25 Reranking")
def bm25_rerank(question, vector_results, top_k=5):
    """
    Rerank FAISS candidate chunks using BM25.

    FAISS provides semantic retrieval.
    BM25 gives additional weight to lexical / keyword
    matches between the question and retrieved chunks.
    """

    if not vector_results:
        return []

    # Tokenize retrieved chunks
    tokenized_documents = [
        result["text"].lower().split()
        for result in vector_results
    ]

    # Tokenize question
    tokenized_question = question.lower().split()

    # Create BM25 model over the FAISS candidates
    bm25 = BM25Okapi(tokenized_documents)

    # Calculate BM25 relevance scores
    scores = bm25.get_scores(tokenized_question)

    # Attach scores to results
    scored_results = []

    for result, score in zip(vector_results, scores):

        scored_result = result.copy()
        scored_result["bm25_score"] = float(score)

        scored_results.append(scored_result)

    # Sort by BM25 score from highest to lowest
    scored_results.sort(
        key=lambda item: item["bm25_score"],
        reverse=True
    )

    # Keep only the best reranked chunks
    return scored_results[:top_k]

# ---------------------------------------------------------
# Cross-Encoder Semantic Reranking
# ---------------------------------------------------------

@traceable(name="Cross-Encoder Reranking")
def cross_encoder_rerank(
    question,
    bm25_results,
    top_k=3
):
    """
    Rerank BM25 results using a cross-encoder.

    The cross-encoder evaluates the question and each
    candidate chunk together, allowing it to capture
    semantic relevance beyond lexical keyword matching.
    """

    if not bm25_results:
        return []

    # Create question/chunk pairs
    pairs = [
        [question, result["text"]]
        for result in bm25_results
    ]

    # Predict semantic relevance scores
    scores = cross_encoder.predict(pairs)

    reranked_results = []

    for result, score in zip(
        bm25_results,
        scores
    ):
        reranked_result = result.copy()

        reranked_result[
            "cross_encoder_score"
        ] = float(score)

        reranked_results.append(
            reranked_result
        )

    # Higher cross-encoder score = more relevant
    reranked_results.sort(
        key=lambda item: item[
            "cross_encoder_score"
        ],
        reverse=True
    )

    return reranked_results[:top_k]
# ---------------------------------------------------------
# Knowledge Graph Retrieval
# ---------------------------------------------------------

@traceable(name="Neo4j Graph Search")
def graph_search(question, limit=10):
    """
    Retrieve graph relationships relevant to the question.

    Candidate relationships are retrieved from Neo4j and
    ranked using simple keyword overlap with the question.
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

    LIMIT 100
    """

    try:
        with driver.session() as session:

            result = session.run(query)

            graph_results = [
                record.data()
                for record in result
            ]

    finally:
        driver.close()

    # -----------------------------------------------------
    # Rank graph relationships by question relevance
    # -----------------------------------------------------

    question_words = set(
        question.lower().split()
    )

    scored_results = []

    for result in graph_results:

        graph_text = (
            f"{result.get('entity', '')} "
            f"{result.get('entity_type', '')} "
            f"{result.get('relationship', '')} "
            f"{result.get('related_entity', '')} "
            f"{result.get('related_type', '')}"
        ).lower()

        score = sum(
            1
            for word in question_words
            if word in graph_text
        )

        if score > 0:

            scored_result = result.copy()
            scored_result["graph_score"] = score

            scored_results.append(
                scored_result
            )

    # Highest relevance first
    scored_results.sort(
        key=lambda item: item["graph_score"],
        reverse=True
    )

    # Return only the most relevant relationships
    return scored_results[:limit]

# ---------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------

@traceable(name="Hybrid RAG Retrieval")
def hybrid_search(
    question,
    vector_k=10,
    bm25_k=5,
    final_k=3
):
    """
    Hybrid retrieval pipeline:

    1. Retrieve candidate chunks using FAISS.
    2. Rerank candidates using BM25.
    3. Rerank BM25 results using a cross-encoder.
    4. Retrieve structured context from Neo4j.
    """

    # -----------------------------------------------------
    # Stage 1: FAISS semantic retrieval
    # -----------------------------------------------------

    vector_candidates = vector_search(
        question,
        k=vector_k
    )

    # -----------------------------------------------------
    # Stage 2: BM25 lexical reranking
    # -----------------------------------------------------

    bm25_results = bm25_rerank(
        question,
        vector_candidates,
        top_k=bm25_k
    )

    # -----------------------------------------------------
    # Stage 3: Cross-encoder semantic reranking
    # -----------------------------------------------------

    vector_results = cross_encoder_rerank(
        question,
        bm25_results,
        top_k=final_k
    )

    # -----------------------------------------------------
    # Knowledge graph retrieval
    # -----------------------------------------------------

    graph_results = graph_search(
        question
    )

    return {
        "question": question,
        "vector_results": vector_results,
        "graph_results": graph_results
    }

# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = "Where are ticket attachments stored?"

    print("\n==============================")
    print("HYBRID RAG TEST")
    print("==============================")

    print("\nQuestion:")
    print(question)

    results = hybrid_search(
        question,
        vector_k=10,
        bm25_k=5,
        final_k=3
  )

    print("\nRetrieval completed.")

    # -----------------------------------------------------
    # Display reranked vector results
    # -----------------------------------------------------

    print("\n==============================")
    print("CROSS-ENCODER RERANKED RESULTS")
    print("==============================")

    for index, result in enumerate(
        results["vector_results"],
        start=1
    ):

        print(
            f"\nResult {index}"
        )

        print(
            f"BM25 Score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Cross-Encoder Score: "
            f"{result['cross_encoder_score']:.4f}"
    )

        print(
            f"FAISS Distance: "
            f"{result['faiss_score']:.4f}"
       )

        print(
            f"File: "
            f"{result['metadata'].get('file_name')}"
        )

        print(
            f"Chunk: "
            f"{result['metadata'].get('chunk_id')}"
        )

        print(
            result["text"][:300]
        )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    answer = generate_answer(
        question,
        results["vector_results"],
        results["graph_results"]
    )

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(answer)