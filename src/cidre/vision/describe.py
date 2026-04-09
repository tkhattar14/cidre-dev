from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from cidre.vision.categorize import parse_categories

IMAGE_PROMPT = """Analyze this image and provide:
Description: A one-sentence description of what the image shows.
Categories: A comma-separated list of categories (e.g., travel, receipt, screenshot, landscape, people, food, document, pet, vehicle, building).
Summary: A brief 1-2 sentence summary.

Respond in exactly this format:
Description: <description>
Categories: <categories>
Summary: <summary>"""

DOCUMENT_PROMPT = """Analyze this document text and provide:
Description: A one-sentence description of what the document is about.
Categories: A comma-separated list of categories (e.g., receipt, invoice, insurance, legal, medical, financial, personal, work).
Summary: A brief 1-2 sentence summary.

Document text:
{text}

Respond in exactly this format:
Description: <description>
Categories: <categories>
Summary: <summary>"""

MARKDOWN_PROMPT = """Analyze this markdown content and provide:
Description: A one-sentence description of what this note is about.
Categories: A comma-separated list of categories (e.g., notes, journal, work, personal, research, project, meeting).
Summary: A brief 1-2 sentence summary.

Content:
{text}

Respond in exactly this format:
Description: <description>
Categories: <categories>
Summary: <summary>"""


def _parse_response(raw: str) -> dict:
    result = {"description": "", "categories": [], "summary": ""}
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("description:"):
            result["description"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("categories:"):
            result["categories"] = parse_categories(line.split(":", 1)[1].strip())
        elif line.lower().startswith("summary:"):
            result["summary"] = line.split(":", 1)[1].strip()
    return result


def describe_image(llm, image_path: Path) -> dict:
    raw = llm.generate_with_image(IMAGE_PROMPT, image_path)
    return _parse_response(raw)


def describe_document(llm, text: str) -> dict:
    prompt = DOCUMENT_PROMPT.format(text=text[:4000])
    raw = llm.generate(prompt)
    return _parse_response(raw)


def describe_markdown(llm, text: str) -> dict:
    prompt = MARKDOWN_PROMPT.format(text=text[:4000])
    raw = llm.generate(prompt)
    return _parse_response(raw)


def describe_video(llm, video_path: Path, frames_dir: Path | None = None) -> dict:
    if frames_dir is None:
        frames_dir = Path(tempfile.mkdtemp())

    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vf", "fps=1/10", "-frames:v", "10",
         str(frames_dir / "frame_%03d.jpg")],
        capture_output=True, check=False,
    )

    frame_descriptions = []
    for frame_path in sorted(frames_dir.glob("frame_*.jpg")):
        desc = describe_image(llm, frame_path)
        frame_descriptions.append(desc["description"])

    if not frame_descriptions:
        return {"description": "Video file (could not extract frames)", "categories": ["video"], "summary": ""}

    combined = " | ".join(frame_descriptions)
    prompt = f"""These are descriptions of frames from a video, taken every 10 seconds:
{combined}

Provide a unified description:
Description: <one sentence describing the overall video>
Categories: <comma-separated categories>
Summary: <1-2 sentence summary>"""

    raw = llm.generate(prompt)
    return _parse_response(raw)
