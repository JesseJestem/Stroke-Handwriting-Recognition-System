from handwriting.core.config import AppConfig
from handwriting.core.paths import (
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#1 App config use default project paths
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_app_config_uses_default_project_paths():
    config = AppConfig()

    assert config.raw_data_dir == RAW_DATA_DIR
    assert config.processed_data_dir == PROCESSED_DATA_DIR
    assert config.outputs_dir == OUTPUTS_DIR
    assert config.saved_models_dir == SAVED_MODELS_DIR

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2 App config use default project paths
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_app_config_allows_path_override(tmp_path):
    custom_raw_dir = tmp_path / "raw"

    config = AppConfig(
        raw_data_dir=custom_raw_dir,
    )

    assert config.raw_data_dir == custom_raw_dir
    assert config.outputs_dir == OUTPUTS_DIR

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3 App config does not create directories
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_app_config_does_not_create_directories(tmp_path):
    custom_outputs_dir = tmp_path / "outputs"

    AppConfig(outputs_dir=custom_outputs_dir)

    assert not custom_outputs_dir.exists()