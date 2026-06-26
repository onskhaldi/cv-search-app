"""
Development validation script for the CV search pipeline.

This is NOT a pytest test file.
Formal automated tests live in tests/ and run with:

    python -m pytest -v

Use this script manually when you want to inspect the full pipeline:
PDF extraction -> chunking -> vector indexing -> retrieval.
"""

import argparse
from pathlib import Path

from app.pdf_reader import extract_text_from_pdf
from app.chunker import chunk_text
from app.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Manually validate the CV search pipeline.")
    parser.add_argument(
        "--pdf",
        default="uploads/cv.pdf",
        help="Path to the PDF file to inspect. Default: uploads/cv.pdf",
    )
    parser.add_argument(
        "--question",
        default="Programmiersprachen Python SQL Java Kotlin C R Assembly",
        help="Search query to run against the CV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=650,
        help="Chunk size for manual validation. Default: 650, matching production settings.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Chunk overlap for manual validation. Default: 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to print. Default: 5.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    results = vector_store.search(args.question, top_k=args.top_k)

    print("Question:", args.question)
    print("PDF:", pdf_path)
    print("Chunks created:", len(chunks))
    print("\nRelevant chunks:\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Chunk {i} ---")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])
        print()


if __name__ == "__main__":
    main()
