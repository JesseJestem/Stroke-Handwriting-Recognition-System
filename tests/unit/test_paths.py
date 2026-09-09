from handwriting.core.paths import (
    DATA_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#1 Absolute project root
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_project_root_is_absolute():
    assert PROJECT_ROOT.is_absolute()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2 Raw/processed is in data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_raw_and_processed_dirs_are_in_data_dir():
    assert RAW_DATA_DIR == DATA_DIR / "raw"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3 Generated paths is in project root
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_generated_paths_are_in_project_root():
    assert OUTPUTS_DIR == PROJECT_ROOT / "outputs"
    assert SAVED_MODELS_DIR == PROJECT_ROOT / "saved_models"