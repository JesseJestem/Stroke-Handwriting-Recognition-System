from pathlib import Path

from handwriting.data.repository import DatasetRepository


def test_list_stroke_files_finds_json_files_recursively(
    tmp_path: Path,
) -> None:
    lower_dir = tmp_path / "strokes" / "lower_a"
    upper_dir = tmp_path / "strokes" / "upper_A"

    lower_dir.mkdir(parents=True)
    upper_dir.mkdir(parents=True)

    lower_file = lower_dir / "a_1.json"
    upper_file = upper_dir / "A_1.json"

    lower_file.write_text("{}", encoding="utf-8")
    upper_file.write_text("{}", encoding="utf-8")

    repository = DatasetRepository(tmp_path)

    result = repository.list_stroke_files()

    assert result == sorted([lower_file, upper_file])


def test_list_stroke_files_finds_only_json_files(
    tmp_path: Path,
) -> None:
    lower_dir = tmp_path / "strokes" / "lower_a"

    lower_dir.mkdir(parents=True)

    lower_file_json = lower_dir / "sample.json"
    lower_file_txt = lower_dir / "note.txt"

    lower_file_json.write_text("{}", encoding="utf-8")
    lower_file_txt.write_text("{}", encoding="utf-8")

    repository = DatasetRepository(tmp_path)

    result = repository.list_stroke_files()

    assert result == sorted([lower_file_json])


def test_list_stroke_files_returns_empty_list_when_no_json_files(
    tmp_path: Path,
) -> None:
    strokes_dir = tmp_path / "strokes"
    strokes_dir.mkdir()

    repository = DatasetRepository(tmp_path)

    result = repository.list_stroke_files()

    assert result == []


def test_load_stroke_returns_stroke_data(
    tmp_path: Path,
) -> None:
    stroke_file = tmp_path / "sample.json"

    stroke_file.write_text(
        """
        {
            "strokes": [
                {
                    "x": 10.0,
                    "y": 20.0,
                    "t": 0.0
                }
            ],
            "label": "a"
        }
        """,
        encoding="utf-8",
    )

    repository = DatasetRepository(tmp_path)

    result = repository.load_stroke(stroke_file)

    assert result["label"] == "a"
    assert result["strokes"][0]["x"] == 10.0