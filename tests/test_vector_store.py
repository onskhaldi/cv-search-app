import sys
import types

import numpy as np


class FakeSentenceTransformer:
    """
    Fake replacement for SentenceTransformer.

    This prevents tests from downloading or loading a real ML model.
    It also makes the retrieval behavior deterministic.
    """

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, *args, **kwargs):
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


fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = fake_sentence_transformers

from app.vector_store import VectorStore


def create_test_vector_store():
    return VectorStore()


def test_search_on_empty_store_returns_empty_list():
    vs = create_test_vector_store()

    result = vs.search("Python")

    assert result == []


def test_build_index_and_search_returns_results():
    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EXPERIENCE\nWorked at TechCorp as backend developer",
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=3)

    assert len(results) > 0


def test_search_returns_dictionaries_with_expected_keys():
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
    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EXPERIENCE\nWorked at TechCorp as backend developer",
    ]

    vs.build_index(chunks)
    results = vs.search("Docker", top_k=1)

    assert isinstance(results[0]["score"], float)


def test_skills_query_returns_skills_chunk():
    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker Kubernetes machine learning",
        "SECTION: HOBBIES\nReading hiking football",
    ]

    vs.build_index(chunks)
    results = vs.search("programming skills Python SQL", top_k=1)

    assert len(results) == 1
    assert "Python" in results[0]["text"]


def test_top_k_limits_results():
    vs = create_test_vector_store()

    chunks = [
        f"SECTION: SKILLS\nPython SQL Docker topic {i}"
        for i in range(10)
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=3)

    assert len(results) <= 3


def test_top_k_larger_than_index_does_not_crash():
    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EDUCATION\nBachelor Computer Science",
    ]

    vs.build_index(chunks)
    results = vs.search("Python", top_k=100)

    assert len(results) <= len(chunks)


def test_rebuilding_index_replaces_previous_content():
    vs = create_test_vector_store()

    vs.build_index(["SECTION: SKILLS\nPython SQL"])
    first_results = vs.search("Python", top_k=1)

    assert len(first_results) == 1
    assert "Python" in first_results[0]["text"]

    vs.build_index(["SECTION: HOBBIES\nReading and hiking"])
    second_results = vs.search("Python", top_k=1)

    assert len(second_results) == 1
    assert "Python" not in second_results[0]["text"]


def test_multilingual_query_expansion_finds_english_skills_chunk():
    vs = create_test_vector_store()

    chunks = [
        "SECTION: SKILLS\nPython SQL Docker",
        "SECTION: EDUCATION\nBachelor Informatik TU Dortmund",
    ]

    vs.build_index(chunks)
    results = vs.search("Welche Programmierkenntnisse hat der Kandidat?", top_k=1)

    assert len(results) == 1
    assert "Python" in results[0]["text"]
