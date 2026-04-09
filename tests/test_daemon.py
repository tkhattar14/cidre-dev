from unittest.mock import MagicMock
from cidre.indexer.daemon import CidreEventHandler, generate_launchd_plist


def test_event_handler_queues_new_file(tmp_path):
    mock_callback = MagicMock()
    handler = CidreEventHandler(
        callback=mock_callback,
        exclude_patterns=["*.py"],
    )

    class FakeEvent:
        src_path = str(tmp_path / "photo.jpg")
        is_directory = False

    handler.on_created(FakeEvent())
    assert mock_callback.call_count == 1
    assert "photo.jpg" in mock_callback.call_args[0][0]


def test_event_handler_ignores_excluded_file(tmp_path):
    mock_callback = MagicMock()
    handler = CidreEventHandler(
        callback=mock_callback,
        exclude_patterns=["*.py"],
    )

    class FakeEvent:
        src_path = str(tmp_path / "code.py")
        is_directory = False

    handler.on_created(FakeEvent())
    assert mock_callback.call_count == 0


def test_generate_launchd_plist():
    plist = generate_launchd_plist()
    assert "com.cidre.watcher" in plist
    assert "cidre" in plist
