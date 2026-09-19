from pathlib import Path
from uuid import uuid4

import pymupdf
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx"
}


def create_document_id() -> str:
    """Create a unique ID for each uploaded document."""
    return str(uuid4())


def get_text_splitter():
    """Create the text splitter used for document chunking."""

    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


def load_pdf(file_path: str, document_id: str):
    """Extract text from PDF while preserving page numbers."""

    documents = []

    pdf = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf, start=1):

        page_text = page.get_text()

        if not page_text.strip():
            continue

        documents.append({
            "text": page_text,
            "metadata": {
                "document_id": document_id,
                "file_name": Path(file_path).name,
                "file_type": ".pdf",
                "page_number": page_number
            }
        })

    pdf.close()

    return documents


def load_text(file_path: str, document_id: str):
    """Load TXT or Markdown file."""

    text = Path(file_path).read_text(
        encoding="utf-8"
    )

    return [{
        "text": text,
        "metadata": {
            "document_id": document_id,
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower()
        }
    }]


def load_docx(file_path: str, document_id: str):
    """Extract text from a DOCX file."""

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    text = "\n".join(paragraphs)

    return [{
        "text": text,
        "metadata": {
            "document_id": document_id,
            "file_name": Path(file_path).name,
            "file_type": ".docx"
        }
    }]


def extract_documents(file_path: str, document_id: str):
    """Extract document content based on file type."""

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        return load_pdf(file_path, document_id)

    if extension in {".txt", ".md"}:
        return load_text(file_path, document_id)

    if extension == ".docx":
        return load_docx(file_path, document_id)

    raise ValueError(
        f"No loader available for {extension}"
    )


def create_chunks(
    documents,
    chunk_size=1000,
    chunk_overlap=150
):
    """Split extracted documents into chunks while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    chunk_id = 1

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        split_texts = splitter.split_text(text)

        for chunk_text in split_texts:

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_id"] = chunk_id

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

            chunk_id += 1

    return chunks


def process_file(file_path: str):
    """Complete ingestion process."""

    document_id = create_document_id()

    documents = extract_documents(
        file_path,
        document_id
    )

    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError(
            "No text could be extracted from the file."
        )

    return {
        "document_id": document_id,
        "file_name": Path(file_path).name,
        "file_type": Path(file_path).suffix.lower(),
        "chunks": chunks,
        "chunk_count": len(chunks)
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    file_path = "data/uploads/spotify_web_app_architecture.pdf"

    try:

        result = process_file(file_path)

        print("\n==============================")
        print("INGESTION TEST")
        print("==============================")

        print(f"Document ID: {result['document_id']}")
        print(f"File name: {result['file_name']}")
        print(f"File type: {result['file_type']}")
        print(f"Number of chunks: {result['chunk_count']}")

        print("\n==============================")
        print("FIRST 3 CHUNKS")
        print("==============================")

        for chunk in result["chunks"][:3]:

            print("\n--- Chunk ---")

            print("Chunk ID:",
                  chunk["metadata"]["chunk_id"])

            print("Page:",
                  chunk["metadata"].get("page_number", "N/A"))

            print("Text:")
            print(chunk["text"])

    except Exception as e:

        print("\n❌ INGESTION FAILED")
        print(f"Error: {e}")