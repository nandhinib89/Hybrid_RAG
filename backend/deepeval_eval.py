from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from backend.retrieval import hybrid_search
from backend.llm import generate_answer


# =========================================================
# Test Questions
# =========================================================

TEST_QUESTIONS = [
    "Where are ticket attachments stored?",
    "What database stores ticket records?",
    "Which service consumes ticket events from RabbitMQ?",
    "What is used for full-text search?",
    "How are customer passwords stored?",
]


# =========================================================
# Main Evaluation
# =========================================================

def main():

    deepeval_cases = []

    # =====================================================
    # Run test cases through actual Hybrid RAG pipeline
    # =====================================================

    for question in TEST_QUESTIONS:

        print("\n" + "=" * 80)
        print("TEST")
        print("=" * 80)

        print(f"\nQuestion:\n{question}")

        # -------------------------------------------------
        # Hybrid retrieval
        # -------------------------------------------------

        results = hybrid_search(
            question,
            vector_k=10,
            bm25_k=5,
            final_k=3
        )

        vector_results = results["vector_results"]
        graph_results = results["graph_results"]

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = generate_answer(
            question,
            vector_results,
            graph_results,
        )

        print(f"\nAnswer:\n{answer}")

        # -------------------------------------------------
        # DeepEval retrieval context
        #
        # Use the final cross-encoder-reranked chunks
        # supplied to the LLM.
        # -------------------------------------------------

        retrieval_context = [
            result["text"]
            for result in vector_results
   ]

        print(
            f"\nRetrieved chunks: "
            f"{len(retrieval_context)}"
        )

        # -------------------------------------------------
        # Retrieval diagnostics
        # -------------------------------------------------

        print("\n" + "=" * 80)
        print("RETRIEVAL DETAILS")
        print("=" * 80)

        for i, result in enumerate(
            vector_results,
            start=1,
        ):

            print(
                f"\n--- Retrieved Chunk {i} ---"
            )

            print(
                f"Chunk ID: "
                f"{result['metadata'].get('chunk_id', 'N/A')}"
            )

            print(
                f"File: "
                f"{result['metadata'].get('file_name', 'N/A')}"
            )

            print(
                f"FAISS Distance: "
                f"{result.get('faiss_score', 0):.4f}"
            )

            print(
                f"BM25 Score: "
                f"{result.get('bm25_score', 0):.4f}"
            )

            print(
                f"Cross-Encoder Score: "
                f"{result.get('cross_encoder_score', 0):.4f}"
            )

            print("\nText:")
            print(result["text"])

        # -------------------------------------------------
        # DeepEval test case
        #
        # IMPORTANT:
        # This must remain inside the question loop so
        # every question becomes a separate test case.
        # -------------------------------------------------

        deepeval_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=retrieval_context,
            )
        )

    # =====================================================
    # DeepEval Metrics
    # =====================================================

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
    )

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
    )

    # =====================================================
    # Run DeepEval
    # =====================================================

    print("\n\n")
    print("=" * 80)
    print("RUNNING DEEPEVAL")
    print("=" * 80)

    evaluate(
        deepeval_cases,
        metrics=[
            answer_relevancy,
            contextual_relevancy,
        ],
    )


if __name__ == "__main__":
    main()