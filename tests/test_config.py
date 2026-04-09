from cidre.config import CidreConfig, load_config, save_config


def test_default_config():
    config = CidreConfig()
    assert config.llm_model == "gemma4"
    assert config.embedding_provider == "ollama"
    assert config.embedding_model == "embeddinggemma"
    assert config.embedding_dimensions == 768
    assert config.sources_watched == []
    assert config.photos_enabled is False
    assert config.exclude_patterns == [
        "*.py", "*.js", "*.ts", "*.swift", "*.go", "*.rs",
        "*.java", "*.c", "*.cpp", "*.h",
        "node_modules", ".git", ".venv", "__pycache__", "Library", ".Trash",
    ]


def test_save_and_load_config(tmp_cidre_home):
    config = CidreConfig()
    config.sources_watched = ["~/Documents", "~/Desktop"]
    config.photos_enabled = True
    save_config(config, tmp_cidre_home / "config.toml")

    loaded = load_config(tmp_cidre_home / "config.toml")
    assert loaded.sources_watched == ["~/Documents", "~/Desktop"]
    assert loaded.photos_enabled is True
    assert loaded.llm_model == "gemma4"


def test_load_missing_config_returns_default(tmp_cidre_home):
    loaded = load_config(tmp_cidre_home / "config.toml")
    assert loaded.llm_model == "gemma4"
