import pytest
from pathlib import Path


@pytest.fixture
def tmp_cidre_home(tmp_path):
    """Provides a temporary ~/.cidre directory for testing."""
    cidre_home = tmp_path / ".cidre"
    cidre_home.mkdir()
    return cidre_home


@pytest.fixture
def sample_files(tmp_path):
    """Creates sample files for indexing tests."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Meeting Notes\nDiscussed project timeline.")
    (docs / "readme.txt").write_text("Project readme content.")

    photos = tmp_path / "photos"
    photos.mkdir()

    return tmp_path
