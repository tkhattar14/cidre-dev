from unittest.mock import patch, MagicMock
from cidre.indexer.photos import list_apple_photos, PhotoInfo


def test_photo_info_dataclass():
    info = PhotoInfo(
        uuid="ABC-123",
        original_path="/path/to/photo.jpg",
        date="2024-06-15T10:30:00",
        title="Beach sunset",
    )
    assert info.uuid == "ABC-123"
    assert info.original_path == "/path/to/photo.jpg"


@patch("cidre.indexer.photos.osxphotos.PhotosDB")
def test_list_apple_photos_returns_photo_infos(mock_db_cls):
    mock_photo = MagicMock()
    mock_photo.uuid = "uuid-1"
    mock_photo.path = "/path/to/IMG_001.jpg"
    mock_photo.date = MagicMock()
    mock_photo.date.isoformat.return_value = "2024-06-15T10:30:00"
    mock_photo.title = "My Photo"
    mock_photo.ismissing = False

    mock_db = MagicMock()
    mock_db.photos.return_value = [mock_photo]
    mock_db_cls.return_value = mock_db

    photos = list_apple_photos()
    assert len(photos) == 1
    assert photos[0].uuid == "uuid-1"
    assert photos[0].original_path == "/path/to/IMG_001.jpg"
