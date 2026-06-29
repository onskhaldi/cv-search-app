"""
Tests for app.pdf_reader.

These tests verify the basic behavior of the PDF text extraction function:
- missing files should raise an exception
- non-PDF files should raise an exception
- valid PDFs should return text as a string
- scanned image-only PDFs are skipped because OCR is not supported
"""

import pytest

from app.pdf_reader import extract_text_from_pdf


def test_missing_file_raises_exception():
    """
    A missing PDF path should raise an exception.

    This confirms that the PDF reader does not silently ignore invalid
    file paths.
    """

    with pytest.raises(Exception):
        extract_text_from_pdf("this_file_does_not_exist.pdf")


def test_non_pdf_path_raises_exception():
    """
    Passing a non-PDF file should raise an exception.

    In the application flow, non-PDF uploads are already rejected in
    app.main before this function is called. This test checks the lower-level
    PDF reader behavior directly.
    """

    with pytest.raises(Exception):
        extract_text_from_pdf("requirements.txt")


@pytest.mark.skip(reason="Requires a real fixture PDF: tests/fixtures/sample_cv.pdf")
def test_returns_string_type_on_valid_pdf():
    """
    A valid text-based PDF should return extracted text as a string.

    This test is skipped until a real sample PDF fixture is added to:
    tests/fixtures/sample_cv.pdf
    """

    text = extract_text_from_pdf("tests/fixtures/sample_cv.pdf")

    assert isinstance(text, str)


@pytest.mark.skip(reason="Requires scanned-image fixture PDF; OCR is not supported")
def test_scanned_image_pdf_returns_empty_string():
    """
    A scanned image-only PDF should return an empty string.

    The project does not use OCR, so image-only PDFs are not expected
    to produce readable text.
    """

    text = extract_text_from_pdf("tests/fixtures/scanned.pdf")

    assert text == ""