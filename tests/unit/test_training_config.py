from handwriting.training.config import TrainingConfig


def test_training_config_uses_default_values():
    config = TrainingConfig()

    assert config.random_seed == 42
    assert config.batch_size == 32
    assert config.epochs == 70
    assert config.learning_rate == 0.0005
    assert config.augmentation_copies == 5


def test_training_config_allows_partial_override():
    config = TrainingConfig(
        batch_size=16,
        epochs=50,
    )

    assert config.batch_size == 16
    assert config.epochs == 50
    assert config.random_seed == 42
    assert config.learning_rate == 0.0005
    assert config.augmentation_copies == 5