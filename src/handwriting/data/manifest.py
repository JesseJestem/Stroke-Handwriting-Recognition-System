import json
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from handwriting.core.exceptions import DatasetManifestError
from handwriting.core.types import StrokeData
from handwriting.data.repository import DatasetRepository


class DatasetManifestEntry(TypedDict):
    path: str
    label: str
    writer_id: NotRequired[str]
    session_id: NotRequired[str]
    source_dataset: NotRequired[str]


class DatasetManifest(TypedDict):
    dataset_version: str
    samples: list[DatasetManifestEntry]


def create_manifest_entry(
    stroke_path: Path,
    raw_data_dir: Path,
    stroke_data: StrokeData,
) -> DatasetManifestEntry:
    if "label" not in stroke_data:
        raise DatasetManifestError("Stroke data must contain a label")

    relative_path = stroke_path.relative_to(raw_data_dir)

    entry: DatasetManifestEntry = {
        "path": relative_path.as_posix(),
        "label": stroke_data["label"],
    }

    if "writer_id" in stroke_data:
        entry["writer_id"] = stroke_data["writer_id"]
    if "session_id" in stroke_data:
        entry["session_id"] = stroke_data["session_id"]
    if "source_dataset" in stroke_data:
        entry["source_dataset"] = stroke_data["source_dataset"]

    return entry


def create_dataset_manifest(
    repository: DatasetRepository,
    dataset_version: str,
) -> DatasetManifest:
    if not dataset_version:
        raise DatasetManifestError(
            "Dataset version must not be empty"
        )

    entries: list[DatasetManifestEntry] = []

    for stroke_path in repository.list_stroke_files():
        stroke_data = repository.load_stroke(stroke_path)

        entry = create_manifest_entry(
            stroke_path=stroke_path,
            raw_data_dir=repository.raw_data_dir,
            stroke_data=stroke_data,
        )

        entries.append(entry)

    return {
        "dataset_version": dataset_version,
        "samples": entries,
    }


def save_dataset_manifest(
    manifest: DatasetManifest,
    manifest_path: str | Path,
) -> None:
    manifest_path = Path(manifest_path)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )


def validate_manifest_entry(
    entry: object,
) -> DatasetManifestEntry:
    if not isinstance(entry, dict):
        raise DatasetManifestError(
            "Dataset manifest entry must be a dictionary"
        )

    if "path" not in entry:
        raise DatasetManifestError(
            "Dataset manifest entry must contain 'path'"
        )

    if not isinstance(entry["path"], str):
        raise DatasetManifestError(
            "Dataset manifest entry 'path' must be a string"
        )

    if "label" not in entry:
        raise DatasetManifestError(
            "Dataset manifest entry must contain 'label'"
        )

    if not isinstance(entry["label"], str):
        raise DatasetManifestError(
            "Dataset manifest entry 'label' must be a string"
        )

    if "writer_id" in entry and not isinstance(entry["writer_id"], str):
        raise DatasetManifestError(
            "Dataset manifest entry 'writer_id' must be a string"
        )

    if "session_id" in entry and not isinstance(entry["session_id"], str):
        raise DatasetManifestError(
            "Dataset manifest entry 'session_id' must be a string"
        )

    if "source_dataset" in entry and not isinstance(entry["source_dataset"],str,):
        raise DatasetManifestError(
            "Dataset manifest entry 'source_dataset' must be a string"
        )

    return cast(DatasetManifestEntry, entry)


def load_dataset_manifest(
    manifest_path: str | Path,
) -> DatasetManifest:
    manifest_path = Path(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise DatasetManifestError(
            "Dataset manifest must be a dictionary"
        )

    if "dataset_version" not in data:
        raise DatasetManifestError(
            "Dataset manifest must contain 'dataset_version'"
        )

    if not isinstance(data["dataset_version"], str):
        raise DatasetManifestError(
            "'dataset_version' must be a string"
        )

    if not data["dataset_version"]:
        raise DatasetManifestError(
            "'dataset_version' must not be empty"
        )

    if "samples" not in data:
        raise DatasetManifestError(
            "Dataset manifest must contain 'samples'"
        )

    if not isinstance(data["samples"], list):
        raise DatasetManifestError(
            "'samples' must be a list"
        )

    for entry in data["samples"]:
        validate_manifest_entry(entry)

    return cast(DatasetManifest, data)