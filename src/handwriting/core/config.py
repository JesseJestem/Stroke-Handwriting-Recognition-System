from dataclasses import dataclass
from pathlib import Path

from handwriting.core.paths import (
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
)


@dataclass(frozen=True)
class AppConfig:
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR
    saved_models_dir: Path = SAVED_MODELS_DIR