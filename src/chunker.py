import json
from pathlib import Path
from hashlib import md5

INPUT_PATH = Path("data/processed/documents.jsonl")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunk_id(document_id, index):
    """
    Create a stable unique ID for each chunk.
    """

    raw = f"{document_id}_{index}"
    return md5(raw.encode("utf-8")).hexdigest()


def chunk_documents():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_documents = 0
    total_chunks = 0

    print("=" * 40)
    print("WIKIMEDIA DOCUMENT CHUNKING")
    print("=" * 40)

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            document = json.loads(line)

            document_id = document["id"]
            title = document["title"]
            url = document["url"]
            source = document["source"]
            text = document["text"]

            chunks = split_text(text)

            for index, chunk_text in enumerate(chunks):

                chunk = {
                    "chunk_id": create_chunk_id(
                        document_id,
                        index
                    ),
                    "document_id": document_id,
                    "title": title,
                    "text": chunk_text,
                    "url": url,
                    "source": source,
                    "chunk_index": index
                }

                outfile.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

            total_documents += 1

            if total_documents % 1000 == 0:
                print(
                    f"Processed documents: "
                    f"{total_documents:,} | "
                    f"Chunks: {total_chunks:,}"
                )

    print()
    print("=" * 40)
    print("CHUNKING COMPLETE")
    print("=" * 40)
    print(f"Documents processed: {total_documents:,}")
    print(f"Chunks created:      {total_chunks:,}")
    print(f"Output file:         {OUTPUT_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    chunk_documents()