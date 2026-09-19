import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def generate_answer(question, vector_results, graph_results):
    """
    Generate a grounded answer using both vector and
    knowledge graph context.
    """

    # -----------------------------------------------------
    # Format vector context
    # -----------------------------------------------------

    vector_context = []

    for result in vector_results:
        metadata = result["metadata"]

        vector_context.append(
            f"""
Source: {metadata.get("file_name")}
Page: {metadata.get("page_number", "N/A")}
Chunk: {metadata.get("chunk_id", "N/A")}

{result["text"]}
"""
        )

    vector_context = "\n---\n".join(vector_context)

    # -----------------------------------------------------
    # Format graph context
    # -----------------------------------------------------

    graph_context = []

    for result in graph_results:
        graph_context.append(
            f"""
{result['entity']}
({result['entity_type']})
--[{result['relationship']}]-->
{result['related_entity']}
({result['related_type']})
"""
        )

    graph_context = "\n".join(graph_context)

    # -----------------------------------------------------
    # Combined prompt
    # -----------------------------------------------------

    prompt = f"""
You are answering questions about a software architecture.

Answer the user's question using ONLY the supplied context.

You have two sources of evidence:

1. VECTOR CONTEXT
   - Contains text extracted from the original architecture document.
   - Use this for detailed information and source grounding.

2. KNOWLEDGE GRAPH CONTEXT
   - Contains structured entities and relationships extracted
     from the architecture document.
   - Use this to understand explicit relationships between
     services, components, databases, and technologies.

IMPORTANT GROUNDING RULES:

1. Do NOT invent information.

2. Do NOT make assumptions based on general software
   architecture knowledge.

3. Do NOT infer that a database stores a particular table
   unless the supplied context explicitly establishes that
   relationship.

4. If the document says that PostgreSQL is used for relational
   data and separately says that a service has certain core
   tables, do NOT automatically conclude that PostgreSQL
   stores those tables unless the context explicitly says so.

5. Prefer explicit knowledge graph relationships when they
   establish a relationship.

   For example:

   Auth & Identity Service
   --[STORES_DATA_IN]-->
   Sessions

   explicitly establishes that relationship.

6. Clearly distinguish between:
   - facts explicitly stated in the context
   - reasonable interpretation
   - information that is not established by the context

7. If the supplied context does not contain enough information
   to answer the question, respond exactly:

   "I don't have enough information in the uploaded document
   to answer that question."

   Do not use your general knowledge to fill in missing information.

8. When vector and graph context describe the same fact,
   combine them into one clear statement rather than
   repeating the information.

9. Do not treat entity names alone as proof of a relationship.
   A relationship must be supported by the document text or
   an explicit knowledge graph relationship.

10. Keep the answer concise and directly answer the user's
    question.

User question:
{question}

==============================
VECTOR CONTEXT
==============================

{vector_context}

==============================
KNOWLEDGE GRAPH CONTEXT
==============================

{graph_context}

==============================
ANSWER
==============================
"""

    response = llm.invoke(prompt)

    return response.content

