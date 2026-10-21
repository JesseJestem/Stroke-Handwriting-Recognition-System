import json
from pathlib import Path

import pytest

from handwriting.core.exceptions import DatasetManifestError
from handwriting.core.types import StrokeData
from handwriting.data.manifest import (
    create_dataset_manifest,
    create_manifest_entry,
    load_dataset_manifest,
    save_dataset_manifest,
    validate_manifest_entry,
)
from handwriting.data.repository import DatasetRepository


def test_create_manifest_entry_returns_relative_path_and_label(
    tmp_path: Path,
) -> None:
    stroke_path = (
        tmp_path
        / "strokes"
        / "lower_a"
        / "sample.json"
    )

    stroke_data: StrokeData = {
        "strokes": [],
        "label": "a",
    }

    result = create_manifest_entry(
        stroke_path=stroke_path,
        raw_data_dir=tmp_path,
        stroke_data=stroke_data,
    )

    assert result == {
        "path": "strokes/lower_a/sample.json",
        "label": "a",
    }


def test_create_manifest_entry_raises_error_when_label_missing(
    tmp_path: Path,
) -> None:
    stroke_path = (
        tmp_path
        / "strokes"
        / "lower_a"
        / "sample.json"
    )

    stroke_data: StrokeData = {
        "strokes": [],
    }

    with pytest.raises(DatasetManifestError):
        create_manifest_entry(
            stroke_path=stroke_path,
            raw_data_dir=tmp_path,
            stroke_data=stroke_data,
        )


def test_create_dataset_manifest_builds_entries_from_repository(
    tmp_path: Path,
) -> None:
    lower_dir = tmp_path / "strokes" / "lower_a"
    upper_dir = tmp_path / "strokes" / "upper_B"

    lower_dir.mkdir(parents=True)
    upper_dir.mkdir(parents=True)

    lower_file = lower_dir / "a_1.json"
    upper_file = upper_dir / "B_1.json"

    lower_file.write_text(
        """
        {
            "strokes": [],
            "label": "a"
        }
        """,
        encoding="utf-8",
    )

    upper_file.write_text(
        """
        {
            "strokes": [],
            "label": "B"
        }
        """,
        encoding="utf-8",
    )

    repository = DatasetRepository(tmp_path)

    result = create_dataset_manifest(repository)

    assert result == {
        "samples": [
            {
                "path": "strokes/lower_a/a_1.json",
                "label": "a",
            },
            {
                "path": "strokes/upper_B/B_1.json",
                "label": "B",
            },
        ]
    }


def test_save_dataset_manifest_writes_json_file(
    tmp_path: Path,
) -> None:
    manifest = {
        "samples": [
            {
                "path": "strokes/lower_a/a_1.json",
                "label": "a",
            },
            {
                "path": "strokes/lower_b/b_1.json",
                "label": "б",
            },
        ]
    }

    manifest_path = tmp_path / "manifest.json"

    save_dataset_manifest(
        manifest=manifest,
        manifest_path=manifest_path,
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data == manifest


def test_load_dataset_manifest_returns_saved_manifest(
    tmp_path: Path,
) -> None:
    manifest = {
        "samples": [
            {
                "path": "strokes/lower_a/a_1.json",
                "label": "a",
            }
        ]
    }

    manifest_path = tmp_path / "manifest.json"

    save_dataset_manifest(
        manifest=manifest,
        manifest_path=manifest_path,
    )

    result = load_dataset_manifest(manifest_path)

    assert result == manifest


def test_load_dataset_manifest_missing_samples_raises_error(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"

    manifest_path.write_text(
        """
        {
            "name": "dataset"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(DatasetManifestError):
        load_dataset_manifest(manifest_path)


def test_load_dataset_manifest_non_dict_raises_error(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"

    manifest_path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(DatasetManifestError):
        load_dataset_manifest(manifest_path)


def test_load_dataset_manifest_samples_non_list_raises_error(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            {
                "samples": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetManifestError):
        load_dataset_manifest(manifest_path)


@pytest.mark.parametrize(
    "entry",
    [
        "not-a-dict",
        {},
        {"path": 123, "label": "a"},
        {"path": "strokes/a.json"},
        {"path": "strokes/a.json", "label": 123},
    ],
)
def test_validate_manifest_entry_rejects_invalid_entries(
    entry: object,
) -> None:
    with pytest.raises(DatasetManifestError):
        validate_manifest_entry(entry)


def test_validate_manifest_entry_returns_valid_entry() -> None:
    entry = {
        "path": "strokes/lower_a/a_1.json",
        "label": "a",
    }

    result = validate_manifest_entry(entry)

    assert result == entry