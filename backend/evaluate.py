import requests


API_URL = "http://127.0.0.1:8000"

FALLBACK_RESPONSE = (
    "I don't have enough information in the uploaded document "
    "to answer that question."
)


# ---------------------------------------------------------
# Evaluation Dataset
# ---------------------------------------------------------

EVALUATION_QUESTIONS = [

    # =====================================================
    # 1. Factual Retrieval
    # =====================================================

    {
        "category": "Factual Retrieval",
        "question": "Where are ticket attachments stored?",
        "expected": ["Amazon S3"],
        "type": "contains"
    },
    {
        "category": "Factual Retrieval",
        "question": "What database stores ticket records?",
        "expected": ["PostgreSQL"],
        "type": "contains"
    },
    {
        "category": "Factual Retrieval",
        "question": "Which service consumes ticket events from RabbitMQ?",
        "expected": ["Notification Service"],
        "type": "contains"
    },
    {
        "category": "Factual Retrieval",
        "question": "What is used for full-text search?",
        "expected": ["Elasticsearch"],
        "type": "contains"
    },
    {
        "category": "Factual Retrieval",
        "question": "How are customer passwords stored?",
        "expected": [
            "salted hashes",
            "not stored as plain text"
        ],
        "type": "contains_any"
    },

    # =====================================================
    # 2. Paraphrased Retrieval
    # =====================================================

    {
        "category": "Paraphrased Retrieval",
        "question": (
            "Where does the system keep files attached "
            "to customer support tickets?"
        ),
        "expected": ["Amazon S3"],
        "type": "contains"
    },
    {
        "category": "Paraphrased Retrieval",
        "question": (
            "Which technology allows users to search "
            "across tickets and support articles?"
        ),
        "expected": ["Elasticsearch"],
        "type": "contains"
    },
    {
        "category": "Paraphrased Retrieval",
        "question": (
            "Where is active session information kept "
            "for faster access?"
        ),
        "expected": ["Redis"],
        "type": "contains"
    },

    # =====================================================
    # 3. Relationship / Context Reasoning
    # =====================================================

    {
        "category": "Relationship Reasoning",
        "question": (
            "What happens to ticket events after the "
            "Ticket Management Service publishes them?"
        ),
        "expected": [
            "RabbitMQ",
            "Notification Service"
        ],
        "type": "contains_all"
    },
    {
        "category": "Relationship Reasoning",
        "question": (
            "How are PostgreSQL and Elasticsearch used "
            "differently by the Knowledge Base Service?"
        ),
        "expected": [
            "PostgreSQL",
            "Elasticsearch"
        ],
        "type": "contains_all"
    },

    # =====================================================
    # 4. Unsupported Questions
    # =====================================================

    {
        "category": "Unsupported Question",
        "question": (
            "What programming language is the mobile "
            "application written in?"
        ),
        "expected": FALLBACK_RESPONSE,
        "type": "fallback"
    },
    {
        "category": "Unsupported Question",
        "question": (
            "What cloud region hosts the customer "
            "support platform?"
        ),
        "expected": FALLBACK_RESPONSE,
        "type": "fallback"
    },

    # =====================================================
    # 5. False-Premise / Hallucination Resistance
    # =====================================================

    {
        "category": "False Premise",
        "question": (
            "Why does MongoDB store the ticket records?"
        ),
        "expected": FALLBACK_RESPONSE,
        "type": "fallback"
    },
    {
        "category": "False Premise",
        "question": (
            "How does Kafka deliver ticket notifications?"
        ),
        "expected": FALLBACK_RESPONSE,
        "type": "fallback"
    },

    # =====================================================
    # 6. Multi-part Context Synthesis
    # =====================================================

    {
        "category": "Multi-part Question",
        "question": (
            "Where are ticket records and ticket "
            "attachments stored?"
        ),
        "expected": [
            "PostgreSQL",
            "Amazon S3"
        ],
        "type": "contains_all"
    }
]


# ---------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------

def evaluate_answer(answer, expected, evaluation_type):

    answer_lower = answer.lower()

    # -----------------------------------------------------
    # Contains one required value
    # -----------------------------------------------------

    if evaluation_type == "contains":

        return all(
            value.lower() in answer_lower
            for value in expected
        )

    # -----------------------------------------------------
    # Contains any accepted value
    # -----------------------------------------------------

    if evaluation_type == "contains_any":

        return any(
            value.lower() in answer_lower
            for value in expected
        )

    # -----------------------------------------------------
    # Contains all required values
    # -----------------------------------------------------

    if evaluation_type == "contains_all":

        return all(
            value.lower() in answer_lower
            for value in expected
        )

    # -----------------------------------------------------
    # Expected grounding fallback
    # -----------------------------------------------------

    if evaluation_type == "fallback":

        return (
            FALLBACK_RESPONSE.lower()
            in answer_lower
        )

    return False


# ---------------------------------------------------------
# Evaluate One Question
# ---------------------------------------------------------

def evaluate_question(item):

    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": item["question"]
        }
    )

    if response.status_code != 200:

        return {
            "category": item["category"],
            "question": item["question"],
            "expected": item["expected"],
            "status": "ERROR",
            "answer": response.text
        }

    result = response.json()

    answer = result["answer"]

    passed = evaluate_answer(
        answer,
        item["expected"],
        item["type"]
    )

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    return {
        "category": item["category"],
        "question": item["question"],
        "expected": item["expected"],
        "status": status,
        "answer": answer
    }


# ---------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("HYBRID RAG - 15 QUESTION EVALUATION")
    print("=" * 70)

    results = []

    for index, item in enumerate(
        EVALUATION_QUESTIONS,
        start=1
    ):

        print(
            f"\n[{index}/"
            f"{len(EVALUATION_QUESTIONS)}]"
        )

        print(
            f"Category: {item['category']}"
        )

        print("\nQuestion:")
        print(item["question"])

        result = evaluate_question(
            item
        )

        results.append(
            result
        )

        print(
            "\nExpected:",
            result["expected"]
        )

        print(
            "Status:",
            result["status"]
        )

        print(
            "Answer:",
            result["answer"]
        )

        print("-" * 70)

    # -----------------------------------------------------
    # Overall Results
    # -----------------------------------------------------

    passed = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    total = len(results)

    accuracy = (
        passed / total
    ) * 100

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {passed}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.1f}%"
    )

    # -----------------------------------------------------
    # Category Results
    # -----------------------------------------------------

    print("\nCATEGORY BREAKDOWN")
    print("-" * 70)

    categories = []

    for item in EVALUATION_QUESTIONS:

        if item["category"] not in categories:
            categories.append(
                item["category"]
            )

    for category in categories:

        category_results = [
            result
            for result in results
            if result["category"] == category
        ]

        category_passed = sum(
            1
            for result in category_results
            if result["status"] == "PASS"
        )

        print(
            f"{category}: "
            f"{category_passed}/"
            f"{len(category_results)}"
        )


if __name__ == "__main__":
    main()