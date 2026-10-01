import os
import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🔎",
    layout="centered"
)


st.title("🔎 Hybrid RAG Assistant")

st.write(
    "Upload a document and ask questions using "
    "hybrid retrieval with FAISS and Neo4j."
)


# --------------------------------
# Session State
# --------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_info" not in st.session_state:
    st.session_state.document_info = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = None

if "question" not in st.session_state:
    st.session_state.question = ""


# --------------------------------
# Question submission function
# --------------------------------

def ask_question():

    question = st.session_state.question

    if not question.strip():
        return

    # Clear previous result before processing
    # the new question
    st.session_state.answer = None
    st.session_state.sources = None

    try:

        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question
            }
        )

        if response.status_code == 200:

            result = response.json()

            st.session_state.answer = (
                result["answer"]
            )

            st.session_state.sources = (
                result["sources"]
            )

        else:

            st.error(
                f"Question failed: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure Uvicorn is running."
        )


# --------------------------------
# 1. Upload a document
# --------------------------------

st.header("1. Upload a document")


uploaded_file = st.file_uploader(
    "Choose a PDF, TXT, Markdown, or DOCX file",
    type=["pdf", "txt", "md", "docx"]
)


if uploaded_file is not None:

    if st.button("Upload Document"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:

            with st.spinner(
                "Processing document..."
            ):

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files
                )

            if response.status_code == 200:

                result = response.json()

                st.session_state.document_uploaded = True

                st.session_state.document_info = (
                    result["document"]
                )

                # Clear previous answer
                st.session_state.answer = None
                st.session_state.sources = None
                st.session_state.question = ""

                st.success(
                    "Document uploaded and processed successfully!"
                )

            else:

                st.session_state.document_uploaded = False

                st.error(
                    f"Upload failed: {response.text}"
                )

        except requests.exceptions.ConnectionError:

            st.session_state.document_uploaded = False

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure Uvicorn is running."
            )


# --------------------------------
# Continue only after upload
# --------------------------------

if st.session_state.document_uploaded:

    document = st.session_state.document_info


    # --------------------------------
    # Document information
    # --------------------------------

    st.write(
        f"**File:** {document['file_name']}"
    )

    st.write(
        f"**Chunks:** {document['chunk_count']}"
    )


    st.divider()


    # --------------------------------
    # 2. Ask a question
    # --------------------------------

    st.header("2. Ask a question")


    st.text_input(
        "Enter your question",
        key="question",
        placeholder=(
            "Where are ticket attachments stored?"
        ),
        on_change=ask_question
    )


    # --------------------------------
    # Ask button
    # --------------------------------

    if st.button("Ask"):

        ask_question()


    # --------------------------------
    # Show loading message
    # --------------------------------

    if (
        st.session_state.question.strip()
        and st.session_state.answer is None
    ):

        st.info(
            "Searching documents and generating answer..."
        )


    # --------------------------------
    # Answer
    # --------------------------------

    if st.session_state.answer:

        st.divider()

        st.subheader("Answer")

        st.write(
            st.session_state.answer
        )


        # --------------------------------
        # Retrieval information
        # --------------------------------

        sources = st.session_state.sources

        st.caption(
            f"Retrieved "
            f"{sources['vector_results']} "
            f"vector results and "
            f"{sources['graph_results']} "
            f"graph results."
        )
        # --------------------------------
        # Sources
        # --------------------------------

        st.subheader("Sources")

        seen_sources = set()

        for source in sources["documents"]:

            file_name = source["file_name"]
            page_number = source["page_number"]
            chunk_id = source["chunk_id"]

            source_key = (
                file_name,
                page_number,
                chunk_id
            )

            # Skip duplicate source entries
            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            if page_number is not None:

                st.write(
                    f"📄 **{file_name}** — "
                    f"Page {page_number}, "
                    f"Chunk {chunk_id}"
                )

            else:

                st.write(
                    f"📄 **{file_name}** — "
                    f"Chunk {chunk_id}"
                )