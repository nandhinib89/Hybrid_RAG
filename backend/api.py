import os
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from backend.ingestion import process_file
from backend.vector_store import add_document_to_vector_store
from backend.knowledge_graph import add_document_to_knowledge_graph
from backend.retrieval import hybrid_search
from backend.llm import generate_answer


# --------------------------------
# FastAPI application
# --------------------------------

app = FastAPI(
    title="Hybrid RAG API",
    description="Hybrid RAG application using FAISS and Neo4j",
    version="1.0.0"
)


# --------------------------------
# Configuration
# --------------------------------

UPLOAD_DIR = "data/uploads"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx"
}


# --------------------------------
# Request models
# --------------------------------

class QuestionRequest(BaseModel):
    question: str


# --------------------------------
# Basic endpoints
# --------------------------------

@app.get("/")
def root():
    return {
        "message": "Hybrid RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------
# Document upload
# --------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    # Get file extension
    extension = os.path.splitext(
        file.filename
    )[1].lower()

    # Validate file type
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Supported types: "
                f"{', '.join(SUPPORTED_EXTENSIONS)}"
            )
        )

    # Create upload directory
    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    # Create file path
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    # --------------------------------
    # Save uploaded file
    # --------------------------------

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    # --------------------------------
    # Process document
    # --------------------------------

    try:

        processed_result = process_file(
            file_path
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document processing failed: {str(e)}"
            )
        )

    # --------------------------------
    # Add to vector database
    # --------------------------------

    try:

        vector_result = add_document_to_vector_store(
            file_path,
            processed_result
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Vector database ingestion failed: "
                f"{str(e)}"
            )
        )

    # --------------------------------
    # Add to knowledge graph
    # --------------------------------

    try:

        graph_result = add_document_to_knowledge_graph(
            file_path,
            processed_result
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Knowledge graph ingestion failed: "
                f"{str(e)}"
            )
        )

    # --------------------------------
    # Return upload result
    # --------------------------------

    return {

        "message": (
            "Document ingested successfully"
        ),

        "document": {

            "document_id":
                processed_result["document_id"],

            "file_name":
                processed_result["file_name"],

            "file_type":
                processed_result["file_type"],

            "chunk_count":
                processed_result["chunk_count"]
        },

        "vector_store": {

            "status": "updated",

            "chunks":
                vector_result["chunk_count"]
        },

        "knowledge_graph": {

            "status": "updated",

            "nodes_processed":
                graph_result["nodes_processed"],

            "relationships_processed":
                graph_result["relationships_processed"]
        }
    }


# --------------------------------
# Question answering
# --------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # Validate question
    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        # --------------------------------
        # Hybrid retrieval
        # --------------------------------

        results = hybrid_search(
            request.question
        )

        # --------------------------------
        # Generate answer
        # --------------------------------

        answer = generate_answer(

            question=request.question,

            vector_results=results[
                "vector_results"
            ],

            graph_results=results[
                "graph_results"
            ]
        )

        # --------------------------------
        # Prepare vector sources
        # --------------------------------

        vector_sources = []

        for result in results["vector_results"]:

            metadata = result["metadata"]

            vector_sources.append({

                "file_name":
                    metadata.get("file_name"),

                "page_number":
                    metadata.get("page_number"),

                "chunk_id":
                    metadata.get("chunk_id")
            })

        # --------------------------------
        # Return answer
        # --------------------------------

        return {

            "question":
                request.question,

            "answer":
                answer,

            "sources": {

                "vector_results":
                    len(
                        results["vector_results"]
                    ),

                "graph_results":
                    len(
                        results["graph_results"]
                    ),

                "documents":
                    vector_sources
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to answer question: "
                f"{str(e)}"
            )
        )