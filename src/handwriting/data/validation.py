from typing import cast

from handwriting.core.exceptions import StrokeDataError
from handwriting.core.types import StrokeData


def validate_stroke_data(data: object) -> StrokeData:
    if not isinstance(data, dict):
        raise StrokeDataError("Stroke data must be a dictionary")

    if "strokes" not in data:
        raise StrokeDataError("Stroke data must contain 'strokes'")

    if not isinstance(data["strokes"], list):
        raise StrokeDataError("'strokes' must be a list")

    for point in data["strokes"]:
        if not isinstance(point, dict):
            raise StrokeDataError("Stroke point must be a dictionary")

        if "x" not in point:
            raise StrokeDataError("Stroke point must contain 'x'")
        if "y" not in point:
            raise StrokeDataError("Stroke point must contain 'y'")
        if "t" not in point:
            raise StrokeDataError("Stroke point must contain 't'")

        if isinstance(point["x"], bool) or not isinstance(point["x"], (int, float)):
            raise StrokeDataError("Stroke point 'x' must be a number")
        if isinstance(point["y"], bool) or not isinstance(point["y"], (int, float)):
            raise StrokeDataError("Stroke point 'y' must be a number")
        if isinstance(point["t"], bool) or not isinstance(point["t"], (int, float)):
            raise StrokeDataError("Stroke point 't' must be a number")

        if "pressure" in point:
            if isinstance(point["pressure"], bool) or not isinstance(
                    point["pressure"],
                    (int, float)
            ):
                raise StrokeDataError("Stroke point 'pressure' must be a number")
        if "pen_down" in point:
            if not isinstance(point["pen_down"], bool):
                raise StrokeDataError("Stroke point 'pen_down' must be a boolean")

    if "label" in data:
        if not isinstance(data["label"], str):
            raise StrokeDataError("'label' must be a string")
    if "canvas_width" in data:
        if isinstance(data["canvas_width"], bool) or not isinstance(data["canvas_width"], int):
            raise StrokeDataError("'canvas_width' must be an integer")
    if "canvas_height" in data:
        if isinstance(data["canvas_height"], bool) or not isinstance(data["canvas_height"], int):
            raise StrokeDataError("'canvas_height' must be an integer")

    return cast(StrokeData, data)