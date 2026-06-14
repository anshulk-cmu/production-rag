from rag.query import QueryAnalyzer, QueryOptimizer


def test_analyze_basic():
    a = QueryAnalyzer().analyze("What is machine learning?")
    assert a.original == "What is machine learning?"
    assert a.is_question is True
    assert a.intent == "definition"
    assert "machine" in a.keywords


def test_detect_intent_how_to():
    assert QueryAnalyzer()._detect_intent("how to train a model") == "how-to"


def test_suggest_strategy_short_query():
    assert QueryAnalyzer().suggest_strategy("ai") == "expand"


def test_remove_stop_words():
    out = QueryOptimizer().remove_stop_words("what is the model")
    assert "what" not in out and "the" not in out and "model" in out


def test_expand_query_keeps_original_and_caps():
    variants = QueryOptimizer().expand_query("how big")
    assert variants[0] == "how big"
    assert 1 <= len(variants) <= 5


def test_optimize_unknown_strategy_returns_original():
    # Locks the dead-path: analyzer can suggest "semantic", which optimize() does not handle.
    assert QueryOptimizer().optimize("hello world", "semantic") == ["hello world"]
