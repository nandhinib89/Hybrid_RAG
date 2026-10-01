import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    # When imported by FastAPI
    from backend.ingestion import process_file
except ModuleNotFoundError:
    # When running directly:
    # python backend/vector_store.py
    from ingestion import process_file


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

VECTOR_DB_PATH = "data/vector_db"


# --------------------------------------------------
# Add document to Vector Store
# --------------------------------------------------

def add_document_to_vector_store(
    file_path,
    processed_result=None
):
    """
    Process a document and create a fresh FAISS vector
    database containing only the uploaded document.

    If processed_result is provided, it is reused so
    the document is processed only once.

    The application currently supports one active
    document at a time, so each upload replaces the
    previous vector index.
    """

    # --------------------------------------------------
    # Process document
    # --------------------------------------------------

    if processed_result is None:

        result = process_file(
            file_path
        )

    else:

        result = processed_result

    chunks = result["chunks"]

    # --------------------------------------------------
    # Convert chunks to LangChain Documents
    # --------------------------------------------------

    documents = []

    for chunk in chunks:

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
        )

    print(
        f"Preparing {len(documents)} chunks "
        f"for vector storage..."
    )

    # --------------------------------------------------
    # Create embeddings
    # --------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # --------------------------------------------------
    # Create fresh vector database
    #
    # The application supports one active document
    # at a time. A new FAISS index is therefore
    # created for every uploaded document.
    # --------------------------------------------------

    os.makedirs(
        VECTOR_DB_PATH,
        exist_ok=True
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    print(
        "Created fresh vector database "
        "for uploaded document."
    )

    # --------------------------------------------------
    # Save FAISS database
    # --------------------------------------------------

    vector_store.save_local(
        VECTOR_DB_PATH
    )

    print(
        f"Vector database saved to: "
        f"{VECTOR_DB_PATH}"
    )

    return {
        "document_id": result["document_id"],
        "file_name": result["file_name"],
        "file_type": result["file_type"],
        "chunk_count": result["chunk_count"]
    }


# --------------------------------------------------
# Direct execution test
# --------------------------------------------------

if __name__ == "__main__":

    FILE_PATH = (
        "data/uploads/"
        "customer_support_platform_architecture.pdf"
    )

    result = add_document_to_vector_store(
        FILE_PATH
    )

    print("\nVector ingestion completed:")

    print(result)