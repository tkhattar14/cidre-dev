from pathlib import Path
from cidre.indexer.scanner import scan_directory, file_hash, classify_file, should_exclude


def test_classify_photo():
    assert classify_file(Path("photo.jpg")) == "photo"
    assert classify_file(Path("image.png")) == "photo"
    assert classify_file(Path("screenshot.heic")) == "photo"
    assert classify_file(Path("pic.webp")) == "photo"


def test_classify_video():
    assert classify_file(Path("clip.mp4")) == "video"
    assert classify_file(Path("movie.mov")) == "video"


def test_classify_document():
    assert classify_file(Path("invoice.pdf")) == "document"


def test_classify_markdown():
    assert classify_file(Path("notes.md")) == "markdown"


def test_classify_unknown():
    assert classify_file(Path("app.exe")) is None
    assert classify_file(Path("code.py")) is None


def test_should_exclude():
    patterns = ["*.py", "node_modules", ".git"]
    assert should_exclude(Path("src/main.py"), patterns) is True
    assert should_exclude(Path("node_modules/pkg/index.js"), patterns) is True
    assert should_exclude(Path(".git/HEAD"), patterns) is True
    assert should_exclude(Path("docs/notes.md"), patterns) is False
    assert should_exclude(Path("photo.jpg"), patterns) is False


def test_file_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = file_hash(f)
    assert isinstance(h1, str)
    assert len(h1) == 64  # sha256 hex

    f.write_text("different content")
    h2 = file_hash(f)
    assert h1 != h2


def test_scan_directory(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "code.py").write_text("print('hello')")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc.pdf").write_bytes(b"%PDF-1.4")

    results = list(scan_directory(tmp_path, exclude_patterns=["*.py"]))
    paths = [r.path for r in results]

    assert tmp_path / "photo.jpg" in paths
    assert tmp_path / "notes.md" in paths
    assert tmp_path / "sub" / "doc.pdf" in paths
    assert tmp_path / "code.py" not in paths
