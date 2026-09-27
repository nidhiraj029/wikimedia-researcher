import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


INPUT_PATH = Path("data/processed/chunks.jsonl")
OUTPUT_DIR = Path("embeddings")

EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


def load_chunks():
    """Load chunk text and metadata from JSONL."""

    chunks = []

    with open(INPUT_PATH, "r", encoding="utf-8") as file:

        for line in file:

            chunk = json.loads(line)

            chunks.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "url": chunk["url"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            })

    return chunks


def generate_embeddings():

    print("=" * 50)
    print("WIKIMEDIA EMBEDDING GENERATION")
    print("=" * 50)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load chunks
    print("\nLoading chunks...")

    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks):,}")

    # Load embedding model
    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    # Prepare texts
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    print("\nGenerating embeddings...")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("\nEmbedding generation complete.")

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Save embeddings
    print("\nSaving embeddings...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings.astype(np.float32)
    )

    # Save metadata separately
    print("Saving metadata...")

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False
        )

    print()
    print("=" * 50)
    print("EMBEDDINGS COMPLETE")
    print("=" * 50)

    print(f"Chunks:       {len(chunks):,}")
    print(f"Dimensions:   {embeddings.shape[1]}")
    print(f"Embeddings:   {EMBEDDINGS_PATH}")
    print(f"Metadata:     {METADATA_PATH}")

    print("=" * 50)


if __name__ == "__main__":
    generate_embeddings()