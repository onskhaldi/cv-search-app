from app.chunker import chunk_text


def test_empty_string_returns_empty_list():
    result = chunk_text("", chunk_size=650, overlap=0)
    assert result == []


def test_whitespace_only_returns_empty_list():
    result = chunk_text("   \n\n   ", chunk_size=650, overlap=0)
    assert result == []


def test_short_text_produces_at_least_one_chunk():
    text = "SKILLS\nPython, SQL, Docker"

    result = chunk_text(text, chunk_size=650, overlap=0)

    assert len(result) >= 1
    assert any("Python" in chunk for chunk in result)


def test_chunk_contains_section_header():
    text = "SKILLS\nPython SQL Docker Kubernetes"

    result = chunk_text(text, chunk_size=650, overlap=0)

    assert any("SKILLS" in chunk.upper() for chunk in result)


def test_multiple_sections_chunked():
    text = (
        "EXPERIENCE\nWorked at TechCorp as a developer for 3 years.\n"
        "EDUCATION\nBachelor of Computer Science, University of Essen.\n"
        "SKILLS\nPython, SQL, Java, Kotlin, Docker."
    )

    result = chunk_text(text, chunk_size=650, overlap=0)

    assert len(result) >= 1


def test_long_section_is_split_into_multiple_chunks():
    long_content = "Python. " * 200
    text = f"SKILLS\n{long_content}"

    result = chunk_text(text, chunk_size=650, overlap=0)

    assert len(result) > 1


def test_chunks_do_not_exceed_reasonable_size():
    long_content = "word " * 500
    text = f"EDUCATION\n{long_content}"

    result = chunk_text(text, chunk_size=650, overlap=0)

    for chunk in result:
        assert len(chunk) <= 1200, f"Chunk too large: {len(chunk)} chars"


def test_smaller_chunk_size_produces_more_chunks():
    long_content = "experience word " * 100
    text = f"EXPERIENCE\n{long_content}"

    chunks_small = chunk_text(text, chunk_size=200, overlap=0)
    chunks_large = chunk_text(text, chunk_size=1200, overlap=0)

    assert len(chunks_small) >= len(chunks_large)


def test_returns_list_of_strings():
    text = "SKILLS\nPython SQL"

    result = chunk_text(text, chunk_size=650, overlap=0)

    assert isinstance(result, list)
    for chunk in result:
        assert isinstance(chunk, str)
