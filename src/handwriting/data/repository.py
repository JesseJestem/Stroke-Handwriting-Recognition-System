from pathlib import Path

from handwriting.core.types import StrokeData
from handwriting.data.loading import load_stroke_json


class DatasetRepository:
    def __init__(self, raw_data_dir: Path) -> None:
        self.raw_data_dir = raw_data_dir

    def list_stroke_files(self) -> list[Path]:
        stroke_dir = self.raw_data_dir / "strokes"

        return sorted(stroke_dir.rglob("*.json"))

    def load_stroke(self, stroke_path: str | Path) -> StrokeData:
        return load_stroke_json(stroke_path)