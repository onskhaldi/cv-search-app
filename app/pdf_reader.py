from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract machine-readable text from a PDF file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    document = fitz.open(path)

    try:
        text_parts = []

        for page in document:
            text_parts.append(page.get_text("text"))

        return "\n".join(text_parts).strip()

    finally:
        document.close()
