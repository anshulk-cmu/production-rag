from rag.config import MODEL_REGISTRY, RAGSettings, get_settings


def test_all_profiles_present():
    assert set(MODEL_REGISTRY) == {"local", "cloud", "zerogpu", "cpu", "fake"}


def test_fake_profile_uses_no_models():
    s = RAGSettings(profile="fake", _env_file=None)
    assert s.models.embedder is None
    assert s.models.llm is None


def test_local_profile_models():
    s = RAGSettings(profile="local", _env_file=None)
    assert s.models.embedder == "BAAI/bge-m3"
    assert "Llama-3.2-3B" in s.models.llm


def test_profile_from_env(monkeypatch):
    monkeypatch.setenv("RAG_PROFILE", "cpu")
    s = RAGSettings(_env_file=None)
    assert s.profile == "cpu"
    assert s.models.embedder == "BAAI/bge-small-en-v1.5"


def test_hf_token_standard(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_standard")
    s = RAGSettings(_env_file=None)
    assert s.hf_token == "hf_standard"


def test_hf_token_legacy_alias(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF-Token", "hf_legacy")
    s = RAGSettings(_env_file=None)
    assert s.hf_token == "hf_legacy"


def test_get_settings_cached():
    assert get_settings() is get_settings()
