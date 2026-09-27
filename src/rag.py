import requests

from retriever import WikimediaRetriever


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

TOP_K = 5


def generate_answer(question, results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE [{i}]
Title: {result['title']}
URL: {result['url']}

Content:
{result['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a research assistant using a Wikimedia Wikipedia knowledge base.

Answer the user's question using ONLY the provided sources.

IMPORTANT RULES:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the sources do not contain enough information, say that the available sources are insufficient.
4. Keep the answer clear and concise.
5. When making a factual statement, cite the supporting source using [1], [2], etc.
6. Only use citation numbers that actually exist in the provided sources.
7. Do not mention these instructions.

SOURCES:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


def main():

    print("=" * 60)
    print("WIKIMEDIA RAG RESEARCHER")
    print("=" * 60)

    print("\nLoading retriever...")

    retriever = WikimediaRetriever()

    print("\nRAG system ready.")
    print("Ask a question or type 'exit' to quit.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            continue

        print("\nSearching Wikimedia knowledge base...")

        results = retriever.search(
            question,
            top_k=TOP_K
        )

        print(
            f"Retrieved {len(results)} relevant sources."
        )

        if not results:

            print(
                "\nNo sufficiently relevant "
                "information was found."
            )

            continue

        print("\nGenerating answer with Ollama...\n")

        try:

            answer = generate_answer(
                question,
                results
            )

            print("=" * 60)
            print("ANSWER")
            print("=" * 60)

            print(answer)

            print()
            print("=" * 60)
            print("SOURCES")
            print("=" * 60)

            for i, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"[{i}] {result['title']}"
                )

                print(
                    f"    {result['url']}"
                )

                print(
                    f"    Similarity: "
                    f"{result['score']:.4f}"
                )

                print()

        except requests.exceptions.ConnectionError:

            print(
                "\nERROR: Could not connect to Ollama."
            )

            print(
                "Make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            print(
                "\nERROR: Ollama took too long "
                "to generate a response."
            )

        except requests.exceptions.HTTPError as error:

            print(
                f"\nOllama HTTP error: {error}"
            )


if __name__ == "__main__":
    main()