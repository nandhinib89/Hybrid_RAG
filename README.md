# Hybrid RAG Assistant

An end-to-end **Hybrid Retrieval-Augmented Generation (RAG)**
application that combines semantic vector retrieval, lexical reranking,
cross-encoder semantic reranking, and knowledge-graph retrieval to
generate grounded answers from uploaded documents.

The application uses **FastAPI** for the backend, **Streamlit** for the
frontend, **FAISS** for vector search, **Neo4j** for knowledge-graph
retrieval, **OpenAI GPT-4o-mini** for grounded generation, **DeepEval**
for evaluation, and **LangSmith** for observability.

## Live Application

**https://hybridrag.sbs**

The application is containerized with Docker and deployed to a VPS
behind Nginx with HTTPS.

------------------------------------------------------------------------

## Key Features

-   Upload PDF, TXT, Markdown, and DOCX documents
-   Preserve filename, page number, and chunk ID metadata
-   Chunk documents with `chunk_size=500` and `chunk_overlap=75`
-   Generate embeddings with OpenAI `text-embedding-3-small`
-   FAISS semantic retrieval (Top 10 candidates)
-   BM25 lexical reranking (Top 5)
-   Cross-Encoder semantic reranking (Top 3)
-   Neo4j entity and relationship storage
-   Question-aware knowledge-graph retrieval
-   Hybrid vector + graph context
-   GPT-4o-mini grounded answer generation
-   Retrieval and prompt-level grounding guardrails
-   Retrieved source display in Streamlit
-   Deterministic functional evaluation
-   DeepEval Answer Relevancy and Contextual Relevancy evaluation
-   LangSmith tracing and observability
-   FastAPI Swagger documentation
-   Docker Compose deployment
-   VPS hosting with Nginx and HTTPS

------------------------------------------------------------------------

## Architecture

``` text
                         User
                           |
                           v
                  Streamlit Frontend
                           |
                           v
                    FastAPI Backend
                           |
                     User Question
                           |
             +-------------+-------------+
             |                           |
             v                           v
      Vector Retrieval            Knowledge Graph
             |                       Retrieval
             v                           |
      FAISS Semantic Search              v
          Top 10                  Neo4j Relationships
             |                           |
             v                           v
       BM25 Reranking              Question-aware
           Top 5                     Filtering
             |                           |
             v                           |
    Cross-Encoder Reranking              |
           Top 3                         |
             +-------------+-------------+
                           |
                           v
                    Grounded Context
                           |
                           v
                       Guardrails
                           |
                           v
                     GPT-4o-mini
                           |
                           v
                Answer + Source Metadata

          Observability: LangSmith
          Evaluation: Functional Tests + DeepEval
```

------------------------------------------------------------------------

## Document Ingestion

``` text
Upload
  |
  v
Validate File Type
  |
  v
Extract Text
  |
  v
Chunk Document
  |
  +--------------------------+
  |                          |
  v                          v
OpenAI Embeddings       Entity / Relationship
  |                       Extraction
  v                          |
FAISS Vector DB              v
                         Neo4j Graph
```

Supported types: **PDF, TXT, Markdown, DOCX**.

PDF files are processed page by page so page information is retained as
source metadata.

Current chunking configuration:

``` text
chunk_size = 500
chunk_overlap = 75
```

------------------------------------------------------------------------

## Hybrid Retrieval Pipeline

### 1. FAISS Semantic Retrieval

The question is embedded with OpenAI embeddings and FAISS retrieves the
top 10 semantic candidates.

### 2. BM25 Lexical Reranking

BM25 reranks those candidates using lexical/keyword relevance and keeps
the top 5.

### 3. Cross-Encoder Semantic Reranking

The model `cross-encoder/ms-marco-MiniLM-L-6-v2` evaluates each
question/chunk pair jointly and returns the final top 3 vector results.

This helps correct cases where keyword overlap ranks a less-direct chunk
above one that more precisely answers the question.

### 4. Neo4j Knowledge-Graph Retrieval

Entities and relationships extracted during ingestion are stored in
Neo4j. At query time, graph relationships are filtered and ranked for
question relevance instead of sending a fixed set of unrelated graph
relationships to the LLM.

### 5. Context Fusion and Generation

The final vector results and relevant graph relationships are supplied
to GPT-4o-mini. The generation prompt instructs the model to answer only
from supplied evidence.

------------------------------------------------------------------------

## Grounding and Guardrails

The generation layer is instructed to:

-   Use only supplied vector and graph context
-   Avoid unsupported assumptions and invented information
-   Prefer explicit graph relationships when they establish a fact
-   Not treat entity names alone as proof of a relationship
-   Return an insufficient-information response when evidence is
    inadequate

Fallback response:

> I don't have enough information in the uploaded document to answer
> that question.

Guardrail logic is implemented in `backend/guardrails.py`.

------------------------------------------------------------------------

## Technology Stack

**Frontend:** Streamlit, Requests

**Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic

**RAG:** LangChain, OpenAI Embeddings, FAISS, BM25, Sentence
Transformers, Cross-Encoder reranking, Neo4j, GPT-4o-mini

**Document Processing:** PyMuPDF, python-docx, LangChain text splitters

**Evaluation & Observability:** Custom functional evaluation, DeepEval,
LangSmith

**Deployment:** Docker, Docker Compose, GitHub, VPS, Nginx, Let's
Encrypt / Certbot

------------------------------------------------------------------------

## Project Structure

``` text
Hybrid_RAG/
|
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── deepeval_eval.py
│   ├── evaluate.py
│   ├── guardrails.py
│   ├── ingestion.py
│   ├── knowledge_graph.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── vector_store.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
|
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
|
├── data/
│   ├── uploads/
│   └── vector_db/
|
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

The repository also contains component test scripts used during
development to validate vector retrieval, graph retrieval, Neo4j
connectivity, and chunk structure.

------------------------------------------------------------------------

## Environment Configuration

Create `backend/.env` from `.env.example`:

``` text
OPENAI_API_KEY=your_openai_api_key

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Hybrid-RAG
```

Never commit the real `backend/.env`.

------------------------------------------------------------------------

## Setup

### 1. Clone

``` bash
git clone https://github.com/nandhinib89/Hybrid_RAG.git
cd Hybrid_RAG
```

### 2. Create and activate a Python 3.11 virtual environment

``` bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 4. Configure `backend/.env`

Add the OpenAI, Neo4j, and LangSmith values shown above.

------------------------------------------------------------------------

## Running Locally

Start FastAPI from the project root:

``` bash
uvicorn backend.api:app --reload
```

-   Backend: `http://127.0.0.1:8000`
-   Swagger: `http://127.0.0.1:8000/docs`

In another terminal:

``` bash
streamlit run frontend/app.py
```

-   Streamlit: `http://localhost:8501`

------------------------------------------------------------------------

## Running with Docker

``` bash
docker compose up --build
```

Containers:

``` text
hybrid-rag-backend
hybrid-rag-frontend
```

Local endpoints:

-   Frontend: `http://localhost:3000`
-   Backend: `http://localhost:8000`
-   Swagger: `http://localhost:8000/docs`

Background mode:

``` bash
docker compose up --build -d
```

Useful commands:

``` bash
docker compose ps
docker compose logs -f
docker compose down
```

Uploaded documents and the FAISS vector database are mounted as Docker
volumes so they persist outside the backend container.

------------------------------------------------------------------------

## API Endpoints

### `GET /`

``` json
{
  "message": "Hybrid RAG API is running"
}
```

### `GET /health`

``` json
{
  "status": "healthy"
}
```

### `POST /upload`

Accepts PDF, TXT, Markdown, and DOCX files. The document is processed
and its chunks are added to both FAISS and the Neo4j knowledge graph.

### `POST /ask`

Example:

``` json
{
  "question": "Where are ticket attachments stored?"
}
```

The response includes the generated answer, retrieval counts, and source
metadata.

------------------------------------------------------------------------

## Evaluation

The project uses two complementary evaluation approaches.

### Functional Evaluation

Run FastAPI first, then:

``` bash
python -m backend.evaluate
```

Current five-question factual test set:

``` text
Passed: 5/5
Accuracy: 100.0%
```

Example questions include:

-   Where are ticket attachments stored?
-   What database stores ticket records?
-   Which service consumes ticket events from RabbitMQ?
-   What is used for full-text search?
-   How are customer passwords stored?

This is a small manually designed factual test set and is not intended
as a comprehensive RAG benchmark.

### DeepEval

Run:

``` bash
python -m backend.deepeval_eval
```

DeepEval evaluates:

-   Answer Relevancy
-   Contextual Relevancy

Testing produced consistently strong answer relevance. It also exposed
an important limitation: a retrieved chunk can contain the correct
evidence together with unrelated surrounding sentences, lowering
Contextual Relevancy.

A controlled experiment evaluating only the top cross-encoder-ranked
chunk improved Contextual Relevancy, supporting the conclusion that
retrieval locates the correct evidence while chunk-level context can
still contain unnecessary material.

The production configuration remains **Top 3** after cross-encoder
reranking rather than being changed solely to optimize an evaluation
metric.

------------------------------------------------------------------------

## LangSmith Observability

LangSmith tracing is enabled for important retrieval stages:

-   FAISS Vector Search
-   BM25 Reranking
-   Cross-Encoder Reranking
-   Neo4j Graph Search
-   Hybrid RAG Retrieval
-   LangChain/OpenAI generation calls

This provides visibility into retrieval behavior, execution flow, LLM
calls, latency, and intermediate pipeline activity.

------------------------------------------------------------------------

## Data Storage

-   Uploaded documents: `data/uploads/`
-   FAISS index: `data/vector_db/`
-   Knowledge graph: Neo4j / Neo4j Aura

Generated uploads and local FAISS database contents are excluded from
Git tracking.

------------------------------------------------------------------------

## Deployment

``` text
GitHub
   |
   v
VPS
   |
   v
Docker Compose
   |
   +--------------------+
   |                    |
   v                    v
FastAPI              Streamlit
Backend              Frontend
   |                    |
   +----------+---------+
              |
              v
            Nginx
              |
              v
          HTTPS / SSL
              |
              v
       https://hybridrag.sbs
```

Docker provides separate frontend and backend containers. Nginx acts as
the reverse proxy, with HTTPS configured using Let's Encrypt / Certbot.

The deployed application is available at **https://hybridrag.sbs**.

------------------------------------------------------------------------

## Security

Secrets are stored locally in `backend/.env`.

The `.gitignore` excludes environment files, virtual environments,
Python caches, DeepEval cache, macOS metadata, IDE configuration, logs,
uploaded documents, and generated FAISS database files.

`.env.example` documents required configuration without exposing
credentials.

------------------------------------------------------------------------

## Current Limitations

-   Retrieved chunks can contain the correct evidence plus unrelated
    surrounding sentences, reducing Contextual Relevancy.
-   Knowledge-graph question matching uses lightweight relevance
    filtering rather than advanced graph reasoning.
-   The current evaluation dataset is small and manually designed.
-   Uploaded documents are stored locally on the deployment host.
-   End-user authentication and authorization are not currently
    implemented.

------------------------------------------------------------------------

## Future Improvements

-   Contextual compression after cross-encoder reranking
-   More granular or structure-aware chunking
-   Larger and more diverse evaluation datasets
-   Additional RAG evaluation metrics
-   More advanced Neo4j traversal and graph reasoning
-   Better provenance and citation presentation
-   Support for CSV and XLSX
-   Authentication and authorization
-   Production-grade structured monitoring and logging

------------------------------------------------------------------------

## Project Status

-   [x] Multi-format document ingestion
-   [x] Metadata-preserving chunking
-   [x] OpenAI embeddings
-   [x] FAISS vector storage
-   [x] BM25 reranking
-   [x] Cross-Encoder semantic reranking
-   [x] Neo4j knowledge graph
-   [x] Question-aware graph retrieval
-   [x] Hybrid vector + graph retrieval
-   [x] GPT-4o-mini grounded generation
-   [x] Retrieval and grounding guardrails
-   [x] Source display
-   [x] FastAPI backend
-   [x] Streamlit frontend
-   [x] Functional evaluation
-   [x] DeepEval evaluation
-   [x] LangSmith observability
-   [x] Docker backend and frontend
-   [x] Docker Compose
-   [x] GitHub source control
-   [x] VPS deployment
-   [x] Nginx reverse proxy
-   [x] HTTPS / SSL
-   [x] Public deployment at `hybridrag.sbs`

------------------------------------------------------------------------

## Live Demo

**https://hybridrag.sbs**

Upload a supported document and ask questions against the ingested
content using the Hybrid RAG pipeline described above.
