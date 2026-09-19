import requests


API_URL = "http://127.0.0.1:8000"


EVALUATION_QUESTIONS = [
    {
        "question": "Where are ticket attachments stored?",
        "expected": "Amazon S3"
    },
    {
        "question": "What database stores ticket records?",
        "expected": "PostgreSQL"
    },
    {
        "question": "Which service consumes ticket events from RabbitMQ?",
        "expected": "Notification Service"
    },
    {
        "question": "What is used for full-text search?",
        "expected": "Elasticsearch"
    },
    {
    "question": "How are customer passwords stored?",
    "expected": [
        "salted hashes",
        "not stored as plain text"
    ]
    }
]


def evaluate_question(question, expected):
    """Send a question to the RAG API and check the answer."""

    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )

    if response.status_code != 200:
        return {
            "question": question,
            "expected": expected,
            "status": "ERROR",
            "answer": response.text
        }

    result = response.json()

    answer = result["answer"]

    expected_values = expected if isinstance(expected, list) else [expected]

    if any(
        value.lower() in answer.lower()
        for value in expected_values
    ):
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "question": question,
        "expected": expected,
        "status": status,
        "answer": answer
    }


def main():

    print("=" * 70)
    print("HYBRID RAG EVALUATION")
    print("=" * 70)

    results = []

    for item in EVALUATION_QUESTIONS:

        print("\nQuestion:")
        print(item["question"])

        result = evaluate_question(
            item["question"],
            item["expected"]
        )

        results.append(result)

        print("Expected:", result["expected"])
        print("Status:", result["status"])
        print("Answer:", result["answer"])

    passed = sum(
        1 for result in results
        if result["status"] == "PASS"
    )

    total = len(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Passed: {passed}/{total}")

    accuracy = (passed / total) * 100

    print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()