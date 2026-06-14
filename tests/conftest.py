import pytest

CORPUS = {
    "doc_1": "machine learning trains models on data",
    "doc_2": "bm25 ranks documents by term frequency and length",
    "doc_3": "neural networks are layers of weighted connections",
    "doc_4": "reciprocal rank fusion combines ranked lists",
}

GOLD = [
    ("term frequency ranking", {"doc_2"}),
    ("training models on data", {"doc_1"}),
    ("rank fusion", {"doc_4"}),
]


@pytest.fixture
def corpus():
    return dict(CORPUS)


@pytest.fixture
def gold():
    return list(GOLD)
