import json
from pathlib import Path

import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/raw/enwiki_namespace_0_00009.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/documents.jsonl"
)


# ============================================================
# SECTION TEXT EXTRACTION
# ============================================================

def extract_section_text(section):
    """
    Recursively extract readable text from a Wikimedia section.
    """

    texts = []

    if not isinstance(section, dict):
        return texts

    # Section name
    name = section.get("name")

    if name:
        texts.append(f"\n{name}\n")

    # Section parts
    parts = section.get("has_parts", [])

    if isinstance(parts, list):

        for part in parts:

            if not isinstance(part, dict):
                continue

            value = part.get("value")

            if value:
                texts.append(str(value))

            # Recursively handle nested sections
            nested = part.get("has_parts")

            if isinstance(nested, list):

                for child in nested:
                    texts.extend(
                        extract_section_text(child)
                    )

    # Some datasets may contain direct values
    value = section.get("value")

    if value:
        texts.append(str(value))

    return texts


# ============================================================
# ARTICLE TEXT EXTRACTION
# ============================================================

def extract_article_text(sections):
    """
    Convert Wikimedia's structured sections JSON
    into plain readable article text.
    """

    if not sections:
        return ""

    try:

        if isinstance(sections, str):
            sections = json.loads(sections)

    except json.JSONDecodeError:

        return ""

    if not isinstance(sections, list):
        return ""

    texts = []

    for section in sections:

        texts.extend(
            extract_section_text(section)
        )

    return "\n".join(texts).strip()


# ============================================================
# INGESTION
# ============================================================

def ingest_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("========================================")
    print("WIKIMEDIA DATASET INGESTION")
    print("========================================")

    parquet_file = pq.ParquetFile(
        DATASET_PATH
    )

    total_rows = parquet_file.metadata.num_rows
    total_groups = parquet_file.num_row_groups

    print(f"Total rows: {total_rows}")
    print(f"Row groups: {total_groups}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed = 0
    skipped = 0

    # --------------------------------------------------------
    # Process one row group at a time
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as output_file:

        for group_number in range(
            total_groups
        ):

            print(
                f"\nProcessing row group "
                f"{group_number + 1}/{total_groups}"
            )

            table = parquet_file.read_row_group(
                group_number
            )

            records = table.to_pylist()

            for record in records:

                # --------------------------------------------
                # Basic metadata
                # --------------------------------------------

                identifier = record.get(
                    "identifier"
                )

                title = record.get(
                    "name"
                )

                abstract = record.get(
                    "abstract"
                )

                url = record.get(
                    "url"
                )

                # --------------------------------------------
                # Extract article body
                # --------------------------------------------

                article_text = extract_article_text(
                    record.get("sections")
                )

                # --------------------------------------------
                # Skip empty articles
                # --------------------------------------------

                if not article_text.strip():

                    skipped += 1

                    continue

                # --------------------------------------------
                # Create normalized document
                # --------------------------------------------

                document = {

                    "id": str(identifier),

                    "title": title or "",

                    "abstract": abstract or "",

                    "text": article_text,

                    "url": url or "",

                    "source": "Wikimedia English Wikipedia"

                }

                # --------------------------------------------
                # Write JSONL
                # --------------------------------------------

                output_file.write(
                    json.dumps(
                        document,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                processed += 1

            print(
                f"Processed so far: {processed}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("INGESTION COMPLETE")
    print("========================================")

    print(
        f"Documents processed: {processed}"
    )

    print(
        f"Documents skipped: {skipped}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    ingest_dataset()