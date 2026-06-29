"""
Tests for app.vector_store.

These tests verify the behavior of the VectorStore retrieval component:
- searching before indexing returns an empty list
- building an index stores searchable chunks
- search results contain text, score, and chunk_id
- top_k limits the number of returned results
- rebuilding the index replaces previous CV content
- multilingual queries can retrieve relevant English CV chunks

The test uses a fake SentenceTransformer model so tests are deterministic
and do not require downloading a real machine learning model.
"""

import sys
import types

import numpy as np


class FakeSentenceTransformer:
    """
    Fake replacement for SentenceTransformer.

    This prevents tests from downloading or loading a real ML model.
    It also makes retrieval behavior deterministic.

    The fake model maps related text categories to fixed vectors:

    - skills/programming text -> [1.0, 0.0, 0.0]
    - work experience text    -> [0.0, 1.0, 0.0]
    - education text          -> [0.0, 0.0, 1.0]
    - unrelated text          -> [0.3, 0.3, 0.3]
    """

    def __init__(self, *args, **kwargs):
        """
        Match the interface of SentenceTransformer.

        The real SentenceTransformer accepts model names and configuration
        arguments. The fake class ignores them because no real model is loaded.
        """

        pass

    def encode(self, texts, *args, **kwargs):
        """
        Convert text into deterministic fake vectors.

        Args:
            texts: A string or a list of strings.

        Returns:
            numpy.ndarray: Fake embedding vectors.
        """

        if isinstance(texts, str):
            texts = [texts]

        vectors = []

        for text in texts:
            text_lower = text.lower()

            if any(
                word in text_lower
                for word in [
                    "python",
                    "sql",
                    "docker",
                    "programming",
                    "skills",
                    "programmierkenntnisse",
                    "programmiersprachen",
                ]
            ):
                vectors.append([1.0, 0.0, 0.0])

            elif any(
                word in text_lower
                for word in [
                    "experience",
                    "worked",
                    "developer",
                    "techcorp",
                ]
            ):
                vectors.append([0.0, 1.0, 0.0])

            elif any(
                word in text_lower
                for word in [
                    "education",
                    "bachelor",
                    "university",
                    "degree",
                    "informatik",
                ]
            ):
                vectors.append([0.0, 0.0, 1.0])

            else:
                vectors.append([0.3, 0.3, 0.3])

        return np.array(vectors, dtype=float)


# -------------------------------------------------------------------
# Mock sentence_transformers before importing VectorStore
# -------------------------------------------------------------------

# Create a fake sentence_transformers module.
fake_sentence_transformers = types.ModuleType("sentence_transformers")

# Add the fake SentenceTransformer class to the fake module.
fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer

# Register the fake module in sys.modules.
# This means when app.vector_store imports SentenceTransformer,
# it receives FakeSentenceTransformer instead of the real one.
sys.modules["sentence_transformers"] = fake_sentence_transformers


from app.vector_store import VectorStore


# -------------------------------------------------------------------
# Test helper
# -------------------------------------------------------------------

def create_test_vector_store():
    """
    Create a new VectorStore instance for each test.

    This keeps tests isolated from each other.
    """

    return VectorStore()


# -------------------------------------------------------------------
# Empty store behavior
# -------------------------------------------------------------------

def test_search_on_empty_store_returns_empty_list():
    """
    Searching before any CV chunks are indexed should return an empty list.
    """

    vs = create_test_vector_store()

    result = vs.search("Python")

    assert result == []


# -------------------------------------------------------------------
# Index building and basic search
# -------------------------------------------------------------------

def test_build_index_and_search_returns_results():
    """
    After building an index, search should return at least one result.
    """

    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EXPERIENCE\nWorked at TechCorp as backend developer",
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=3)

    assert len(results) > 0


def test_search_returns_dictionaries_with_expected_keys():
    """
    Search results should be dictionaries containing:
    - text: the original chunk text
    - score: similarity score
    - chunk_id: index of the matched chunk
    """

    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EXPERIENCE\nWorked at TechCorp as backend developer",
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=1)

    assert len(results) == 1
    assert "text" in results[0]
    assert "score" in results[0]
    assert "chunk_id" in results[0]


def test_search_score_is_numeric():
    """
    The similarity score returned by search should be a float.
    """

    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EXPERIENCE\nWorked at TechCorp as backend developer",
    ]

    vs.build_index(chunks)
    results = vs.search("Docker", top_k=1)

    assert isinstance(results[0]["score"], float)


# -------------------------------------------------------------------
# Retrieval quality
# -------------------------------------------------------------------

def test_skills_query_returns_skills_chunk():
    """
    A skills-related query should retrieve the skills chunk.
    """

    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker Kubernetes machine learning",
        "SECTION: HOBBIES\nReading hiking football",
    ]

    vs.build_index(chunks)
    results = vs.search("programming skills Python SQL", top_k=1)

    assert len(results) == 1
    assert "Python" in results[0]["text"]


def test_multilingual_query_expansion_finds_english_skills_chunk():
    """
    A German programming-skills query should retrieve the English skills chunk.

    This verifies that multilingual or synonym-style query handling works
    for common CV questions.
    """

    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EDUCATION\nBachelor Informatik TU Dortmund",
    ]

    vs.build_index(chunks)
    results = vs.search("Welche Programmierkenntnisse hat der Kandidat?", top_k=1)

    assert len(results) == 1
    assert "Python" in results[0]["text"]


# -------------------------------------------------------------------
# top_k behavior
# -------------------------------------------------------------------

def test_top_k_limits_results():
    """
    top_k should limit the maximum number of returned search results.
    """

    vs = create_test_vector_store()

    chunks = [
        f"SECTION: SKILLS\nPython SQL Docker topic {i}"
        for i in range(10)
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=3)

    assert len(results) <= 3


def test_top_k_larger_than_index_does_not_crash():
    """
    If top_k is larger than the number of indexed chunks, search should
    not crash and should return at most the number of available chunks.
    """

    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EDUCATION\nBachelor Computer Science",
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=100)

    assert len(results) <= len(chunks)


# -------------------------------------------------------------------
# Rebuilding index
# -------------------------------------------------------------------

def test_rebuilding_index_replaces_previous_content():
    """
    Building a new index should replace the old indexed CV content.

    This is important because the app stores one active CV at a time.
    """

    vs = create_test_vector_store()

    vs.build_index(["SECTION: SKILLS\nPython SQL"])
    first_results = vs.search("Python", top_k=1)

    assert len(first_results) == 1
    assert "Python" in first_results[0]["text"]

    vs.build_index(["SECTION: HOBBIES\nReading and hiking"])
    second_results = vs.search("Python", top_k=1)

    assert len(second_results) == 1
    assert "Python" not in second_results[0]["text"]