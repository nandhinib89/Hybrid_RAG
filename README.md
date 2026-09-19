# Hybrid RAG Assistant

A Hybrid Retrieval-Augmented Generation (RAG) application that combines **vector search using FAISS** and **knowledge graph retrieval using Neo4j** to answer questions about uploaded documents.

The application uses **FastAPI** for the backend, **Streamlit** for the frontend, and **OpenAI GPT-4o-mini** for grounded answer generation.

---

## Features

- Upload PDF, TXT, Markdown, and DOCX documents
- Extract and chunk document content
- Store document chunks in a FAISS vector database
- Extract entities and relationships into a Neo4j knowledge graph
- Perform hybrid retrieval using vector similarity search and knowledge graph retrieval
- Generate grounded answers using GPT-4o-mini
- Display retrieved document sources
- Basic grounding guardrail to reduce unsupported answers
- Lightweight evaluation of RAG responses
- Dockerized backend and frontend
- FastAPI Swagger documentation

---

## Architecture

```text
                    User
                      |
                      v
              Streamlit Frontend
                    :3000
                      |
                      v
                FastAPI Backend
                    :8000
                      |
             +--------+--------+
             |                 |
             v                 v
       Vector Retrieval   Graph Retrieval
             |                 |
             v                 v
           FAISS             Neo4j
             |                 |
             +--------+--------+
                      |
                      v
              Context Fusion
                      |
                      v
                GPT-4o-mini
                      |
                      v
              Grounded Answer
```

---

## Document Ingestion

When a document is uploaded, it follows this pipeline:

```text
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
  +--------------------+
  |                    |
  v                    v
FAISS Vector DB     Neo4j Knowledge Graph
```

Supported document types:

- PDF
- TXT
- Markdown
- DOCX

PDF documents are processed page by page so that page information can be retained as source metadata.

---

## Query Pipeline

When a user asks a question:

```text
Question
   |
   +--------------------+
   |                    |
   v                    v
FAISS Search       Neo4j Retrieval
   |                    |
   +---------+----------+
             |
             v
       Retrieved Context
             |
             v
        GPT-4o-mini
             |
             v
      Grounded Answer
```

The LLM is instructed to use only the retrieved context and avoid inventing information that is not supported by the uploaded documents.

If the supplied context does not contain enough information, the system is instructed to respond:

> "I don't have enough information in the uploaded document to answer that question."

---

## Technology Stack

### Frontend

- Streamlit

### Backend

- FastAPI
- Uvicorn
- Python 3.11

### RAG

- FAISS
- Neo4j
- LangChain
- OpenAI embeddings
- GPT-4o-mini

### Document Processing

- PyMuPDF
- python-docx
- LangChain text splitters

### Deployment

- Docker
- Docker Compose
- GitHub
- VPS
- Nginx
- Let's Encrypt / Certbot

---

## Project Structure

```text
Hybrid_RAG/
│
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── evaluate.py
│   ├── ingestion.py
│   ├── knowledge_graph.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── vector_store.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
│
├── data/
│   ├── uploads/
│   └── vector_db/
│
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

Create `backend/.env` locally using `.env.example`. It contains secrets and is intentionally excluded from GitHub.

---

# Setup

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Hybrid_RAG
```

## 2. Configure environment variables

Create:

```text
backend/.env
```

with:

```text
OPENAI_API_KEY=your_openai_api_key

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

Never commit the real `.env` file to GitHub.

An `.env.example` file is provided as a template.

---

# Running Locally Without Docker

## Start FastAPI

From the project root:

```bash
uvicorn backend.api:app --reload
```

FastAPI:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Start Streamlit

In another terminal:

```bash
streamlit run frontend/app.py
```

Streamlit:

```text
http://localhost:8501
```

---

# Running with Docker

From the project root:

```bash
docker compose up --build
```

The application consists of two containers:

```text
hybrid-rag-backend
hybrid-rag-frontend
```

### Frontend

```text
http://localhost:3000
```

### Backend

```text
http://localhost:8000
```

### FastAPI documentation

```text
http://localhost:8000/docs
```

## Run Docker in the background

```bash
docker compose up --build -d
```

Check containers:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f
```

View backend logs:

```bash
docker compose logs backend
```

Stop:

```bash
docker compose down
```

---

# API Endpoints

## Health Check

```text
GET /
```

Returns:

```json
{
  "message": "Hybrid RAG API is running"
}
```

## Health

```text
GET /health
```

Returns:

```json
{
  "status": "healthy"
}
```

## Upload Document

```text
POST /upload
```

Accepts PDF, TXT, Markdown, and DOCX.

The document is processed and added to both the vector database and knowledge graph.

## Ask a Question

```text
POST /ask
```

Example:

```json
{
  "question": "Where are ticket attachments stored?"
}
```

The response contains the generated answer, retrieval counts, and retrieved document source information.

---

# Evaluation

A lightweight evaluation script is included in:

```text
backend/evaluate.py
```

The evaluation uses five factual questions based on the customer-support architecture document.

Example questions include:

- Where are ticket attachments stored?
- What database stores ticket records?
- Which service consumes ticket events from RabbitMQ?
- What is used for full-text search?
- How are customer passwords stored?

Run:

```bash
python backend/evaluate.py
```

## Evaluation Result

Current evaluation:

```text
Test cases: 5
Passed: 4
Failed: 1
Pass rate: 80%
```

The failed test demonstrated the application's grounding behavior: when the required information was not available in the retrieved context, the system returned the configured insufficient-information response rather than generating an unsupported answer.

This is a small manually designed factual test set and should not be interpreted as a comprehensive RAG benchmark.

---

# Grounding Guardrail

The LLM prompt instructs the model to:

- Use only the supplied document and graph context
- Avoid unsupported assumptions
- Avoid inventing information
- Prefer explicit knowledge graph relationships
- Distinguish established information from unsupported information
- Refuse to answer when the supplied context is insufficient

This provides a lightweight application-level grounding guardrail without requiring an additional guardrail framework.

---

# Data Storage

### FAISS

Vector embeddings for document chunks are stored locally in:

```text
data/vector_db/
```

### Neo4j

Extracted entities and relationships are stored in Neo4j.

The application is configured to use Neo4j Aura through the environment variables in `backend/.env`.

---

# Docker Architecture

```text
                  Browser
                     |
                     v
              localhost:3000
                     |
                     v
        +-------------------------+
        | Streamlit Frontend      |
        | hybrid-rag-frontend     |
        | Port 8501               |
        +------------+------------+
                     |
                     | Docker network
                     |
                     v
        +-------------------------+
        | FastAPI Backend         |
        | hybrid-rag-backend      |
        | Port 8000               |
        +------------+------------+
                     |
              +------+------+
              |             |
              v             v
           FAISS        Neo4j Aura
```

The FAISS database and uploaded documents are mounted as local Docker volumes so that they remain available outside the container.

---

# Deployment

The project is designed to follow this deployment workflow:

```text
Local Application
       |
       v
Docker
       |
       v
GitHub
       |
       v
VPS
       |
       v
Docker Compose
       |
       v
Nginx
       |
       v
Domain
       |
       v
HTTPS / SSL
```

The deployment workflow uses Docker for containerization, GitHub for source-code storage, a VPS for hosting, Nginx as a reverse proxy, and Let's Encrypt / Certbot for HTTPS.

The local Docker deployment should be verified before deploying to the VPS.

---

# Security

The following file must not be committed to GitHub:

```text
backend/.env
```

The `.gitignore` file excludes environment files and other local development artifacts.

Use:

```text
.env.example
```

to document the required environment variables without exposing credentials.

---

# Future Improvements

Potential improvements include:

- More sophisticated evaluation using RAG evaluation frameworks
- Improved graph query relevance
- Reranking of retrieved vector results
- Stronger document-level grounding
- Better document provenance tracking
- Support for CSV and XLSX files
- Improved retrieval evaluation
- Production monitoring and logging
- Authentication and authorization

---

# Project Status

Current implementation includes:

- [x] Multi-format document ingestion
- [x] Document chunking
- [x] FAISS vector storage
- [x] Neo4j knowledge graph
- [x] Hybrid retrieval
- [x] GPT-4o-mini answer generation
- [x] Grounding guardrail
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] Source display
- [x] Evaluation script
- [x] Docker backend
- [x] Docker frontend
- [x] Docker Compose
- [x] Local Docker testing

Next deployment steps:

- [ ] Push project to GitHub
- [ ] Deploy to VPS
- [ ] Configure domain
- [ ] Configure Nginx
- [ ] Enable HTTPS with Certbot
