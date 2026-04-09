from cidre.providers.base import get_provider
from cidre.providers.ollama import OllamaEmbedding, OllamaLLM
from cidre.providers.openai import OpenAIEmbedding


def test_ollama_embedding_properties():
    provider = OllamaEmbedding(model="embeddinggemma")
    assert provider.name == "ollama:embeddinggemma"
    assert provider.dimensions == 768


def test_ollama_embedding_bge_m3():
    provider = OllamaEmbedding(model="bge-m3")
    assert provider.dimensions == 1024


def test_openai_embedding_properties():
    provider = OpenAIEmbedding(model="text-embedding-3-large", api_key="test-key")
    assert provider.name == "openai:text-embedding-3-large"
    assert provider.dimensions == 3072


def test_get_provider_ollama():
    provider = get_provider("ollama", "embeddinggemma")
    assert isinstance(provider, OllamaEmbedding)
    assert provider.dimensions == 768


def test_get_provider_openai():
    provider = get_provider("openai", "text-embedding-3-large", api_key="test")
    assert isinstance(provider, OpenAIEmbedding)


def test_ollama_llm_properties():
    llm = OllamaLLM(model="gemma4:26b-a4b")
    assert llm.model == "gemma4:26b-a4b"
