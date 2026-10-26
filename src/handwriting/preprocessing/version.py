from typing import Final, TypedDict

PREPROCESSING_VERSION: Final[str] = "v1"


class PreprocessingMetadata(TypedDict):
    preprocessing_version: str


def get_preprocessing_metadata() -> PreprocessingMetadata:
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
    }