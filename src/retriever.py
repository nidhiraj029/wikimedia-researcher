import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


METADATA_PATH = Path("embeddings/metadata.json")
INDEX_PATH = Path("embeddings/wikimedia.index")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieve more candidates first, then filter them.
SEARCH_K = 10

# Minimum similarity required to keep a result.
MIN_SIMILARITY = 0.35


def clean_url(url):
    """
    Convert Wikimedia markdown-style URLs into normal URLs.
    """

    if not url:
        return ""

    match = re.search(
        r"\((https?://[^)]+)\)",
        url
    )

    if match:
        return match.group(1)

    if url.startswith("http"):
        return url

    return url


class WikimediaRetriever:

    def __init__(self):

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        print(
            f"FAISS vectors loaded: "
            f"{self.index.ntotal:,}"
        )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding model loaded.")

        print("Loading metadata...")

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        print(
            f"Metadata loaded: "
            f"{len(self.metadata):,}"
        )

    def search(
        self,
        query,
        top_k=5,
        min_similarity=MIN_SIMILARITY
    ):

        # Create query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype(
            np.float32
        )

        # Retrieve candidate results
        scores, indices = self.index.search(
            query_embedding,
            SEARCH_K
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            score = float(score)

            # Filter weak matches
            if score < min_similarity:
                continue

            metadata = self.metadata[index]

            results.append({
                "score": score,
                "chunk_id": metadata["chunk_id"],
                "document_id": metadata["document_id"],
                "title": metadata["title"],
                "text": metadata["text"],
                "url": clean_url(metadata["url"]),
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"]
            })

            if len(results) >= top_k:
                break

        return results


def main():

    print()
    print("=" * 55)
    print("WIKIMEDIA SEMANTIC SEARCH")
    print("=" * 55)

    retriever = WikimediaRetriever()

    print()
    print("Type a question to search Wikipedia.")
    print("Type 'exit' to quit.")
    print()

    while True:

        query = input("Query: ").strip()

        if query.lower() == "exit":
            print("\nExiting...")
            break

        if not query:
            continue

        results = retriever.search(
            query,
            top_k=5
        )

        print()
        print("=" * 55)
        print("SEARCH RESULTS")
        print("=" * 55)

        if not results:

            print(
                "\nNo sufficiently relevant "
                "results were found."
            )

            continue

        for i, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"[{i}] {result['title']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"URL: {result['url']}"
            )

            print(
                f"Text: {result['text'][:500]}..."
            )

        print()
        print("=" * 55)


if __name__ == "__main__":
    main()