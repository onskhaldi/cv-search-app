import pytest

from app.pdf_reader import extract_text_from_pdf


def test_missing_file_raises_exception():
    with pytest.raises(Exception):
        extract_text_from_pdf("this_file_does_not_exist.pdf")


def test_non_pdf_path_raises_exception():
    with pytest.raises(Exception):
        extract_text_from_pdf("requirements.txt")


@pytest.mark.skip(reason="Requires a real fixture PDF: tests/fixtures/sample_cv.pdf")
def test_returns_string_type_on_valid_pdf():
    text = extract_text_from_pdf("tests/fixtures/sample_cv.pdf")

    assert isinstance(text, str)


@pytest.mark.skip(reason="Requires scanned-image fixture PDF; OCR is not supported")
def test_scanned_image_pdf_returns_empty_string():
    text = extract_text_from_pdf("tests/fixtures/scanned.pdf")

    assert text == ""
