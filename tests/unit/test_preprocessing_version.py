from handwriting.preprocessing.version import (
    PREPROCESSING_VERSION,
    get_preprocessing_metadata,
)


def test_get_preprocessing_metadata_returns_current_version() -> None:
    result = get_preprocessing_metadata()

    assert result == {
        "preprocessing_version": PREPROCESSING_VERSION,
    }