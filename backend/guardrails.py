"""
Simple guardrails for the Hybrid RAG application.

The goal is to prevent the LLM from answering when retrieval
does not provide meaningful evidence for the user's question.
"""

import re


FALLBACK_RESPONSE = (
    "I don't have enough information in the uploaded document "
    "to answer that question."
)


def tokenize(text):
    """
    Convert text into normalized word tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def has_relevant_context(
    question,
    vector_results,
    minimum_overlap=1
):
    """
    Check whether retrieved vector context has at least some
    lexical overlap with the user's question.

    BM25 has already reranked the FAISS candidates, so this
    acts only as a lightweight safety check before generation.
    """

    if not vector_results:
        return False

    question_tokens = tokenize(question)

    # Ignore very common question words
    stop_words = {
        "what",
        "where",
        "when",
        "which",
        "who",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "does",
        "do",
        "used"
    }

    meaningful_question_tokens = (
        question_tokens - stop_words
    )

    if not meaningful_question_tokens:
        return False

    for result in vector_results:

        context_tokens = tokenize(
            result["text"]
        )

        overlap = (
            meaningful_question_tokens
            & context_tokens
        )

        if len(overlap) >= minimum_overlap:
            return True

    return False


def validate_retrieval(
    question,
    vector_results
):
    """
    Validate whether retrieved evidence is sufficient
    to proceed to answer generation.

    Returns:
        (allowed, message)
    """

    if not vector_results:

        return False, FALLBACK_RESPONSE

    if not has_relevant_context(
        question,
        vector_results
    ):

        return False, FALLBACK_RESPONSE

    return True, None