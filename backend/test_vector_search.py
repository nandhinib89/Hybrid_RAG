from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


VECTOR_DB_PATH = "data/vector_db"


def search_vector_db(question, k=3):

    # Load the same embedding model used when creating the database
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Load the saved FAISS database
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Search for the most similar chunks
    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results


if __name__ == "__main__":

    question = "What databases are used in the architecture?"

    results = search_vector_db(question)

    print("\n==============================")
    print("VECTOR SEARCH RESULTS")
    print("==============================")

    print(f"\nQuestion: {question}")

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print("Chunk ID:", result.metadata.get("chunk_id"))
        print("Page:", result.metadata.get("page_number", "N/A"))
        print("File:", result.metadata.get("file_name"))

        print("\nText:")
        print(result.page_content)