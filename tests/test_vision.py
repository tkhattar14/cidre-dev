from unittest.mock import MagicMock, patch
from cidre.vision.describe import describe_image, describe_video, describe_document, describe_markdown
from cidre.vision.categorize import parse_categories


def test_describe_image_calls_llm_with_image(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

    mock_llm = MagicMock()
    mock_llm.generate_with_image.return_value = (
        'Description: A sunset over the ocean\n'
        'Categories: travel, landscape, sunset\n'
        'Summary: Beach sunset photo'
    )

    result = describe_image(mock_llm, img)
    mock_llm.generate_with_image.assert_called_once()
    assert "sunset" in result["description"].lower()
    assert isinstance(result["categories"], list)
    assert isinstance(result["summary"], str)


def test_describe_markdown_calls_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        'Description: Meeting notes about project timeline\n'
        'Categories: work, notes\n'
        'Summary: Notes from team meeting'
    )

    result = describe_markdown(mock_llm, "# Meeting\nDiscussed timeline.")
    mock_llm.generate.assert_called_once()
    assert "description" in result
    assert "categories" in result


def test_parse_categories():
    raw = "travel, landscape, sunset"
    result = parse_categories(raw)
    assert result == ["travel", "landscape", "sunset"]


def test_parse_categories_with_brackets():
    raw = "[travel, landscape]"
    result = parse_categories(raw)
    assert result == ["travel", "landscape"]


def test_parse_categories_empty():
    assert parse_categories("") == []
    assert parse_categories("none") == []
