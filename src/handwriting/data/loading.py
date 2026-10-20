import json
from pathlib import Path

from handwriting.core.exceptions import StrokeDataError
from handwriting.core.types import StrokeData
from handwriting.data.validation import validate_stroke_data


def load_stroke_json(stroke_path: str | Path) -> StrokeData:
    stroke_path = Path(stroke_path)

    try:
        with open(stroke_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except FileNotFoundError as exc:
        raise StrokeDataError(
            f"Stroke file not found: {stroke_path}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise StrokeDataError(
            f"Invalid stroke JSON: {stroke_path}"
        ) from exc

    return validate_stroke_data(data)