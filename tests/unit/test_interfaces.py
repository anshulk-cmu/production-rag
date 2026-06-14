import numpy as np

from rag.interfaces import (
    Chunk,
    Chunker,
    Document,
    DocumentLoader,
    Embedder,
    GenerationResult,
    Generator,
    GradeResult,
    LLMClient,
    QueryTransform,
    Reranker,
    RetrievalGrader,
    Router,
    RouteDecision,
    VectorStore,
)


def test_embedder_protocol_positive():
    class E:
        dim = 3

        def embed(self, text):
            return np.zeros(3)

        def embed_batch(self, texts, batch_size=32):
            return np.zeros((len(texts), 3))

    assert isinstance(E(), Embedder)


def test_embedder_protocol_negative():
    class Bad:
        dim = 3

    assert not isinstance(Bad(), Embedder)


def test_reranker_protocol():
    class R:
        def score(self, q, d):
            return 0.0

        def score_batch(self, q, ds):
            return [0.0] * len(ds)

    assert isinstance(R(), Reranker)


def test_vectorstore_protocol():
    class V:
        dim = 3

        def add(self, ids, vectors, metadata=None): ...

        def search(self, qv, k):
            return []

        def save(self, path): ...

        def load(self, path): ...

        def __len__(self):
            return 0

    assert isinstance(V(), VectorStore)


def test_llmclient_protocol():
    class L:
        def generate(self, prompt, *, max_new_tokens=512, temperature=0.0, stop=None):
            return ""

        def stream(self, prompt, **kw):
            return iter(())

    assert isinstance(L(), LLMClient)


def test_chunker_and_loader_protocols():
    class C:
        def chunk(self, text, doc_id, metadata=None):
            return []

    class D:
        def load(self, source):
            return []

    assert isinstance(C(), Chunker)
    assert isinstance(D(), DocumentLoader)


def test_transform_grader_router_generator_protocols():
    class T:
        def transform(self, q):
            return [q]

    class G:
        def grade(self, q, contexts):
            return GradeResult("correct", 1.0, "use")

    class Ro:
        def route(self, q):
            return RouteDecision("hybrid", "hybrid")

    class Gen:
        def generate(self, q, contexts):
            return GenerationResult("ans")

    assert isinstance(T(), QueryTransform)
    assert isinstance(G(), RetrievalGrader)
    assert isinstance(Ro(), Router)
    assert isinstance(Gen(), Generator)


def test_dataclasses_construct():
    assert Document("d1", "text").doc_id == "d1"
    assert Chunk("c1", "d1", "t").chunk_id == "c1"
    assert GradeResult("correct", 0.9, "use").label == "correct"
    assert RouteDecision("semantic", "semantic").strategy == "semantic"
    gr = GenerationResult("answer")
    assert gr.answer == "answer" and gr.citations == []
