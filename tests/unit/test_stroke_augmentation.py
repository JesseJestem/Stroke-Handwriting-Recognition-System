import logging

import numpy as np

from handwriting.training.augmentation import (
    augment_strokes,
    augment_training_data,
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#TEST SAMPLES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

strokes = np.array([
    [0.20, 0.30, 0.00, 0.5, 1.0, 1.0],
    [0.30, 0.40, 0.25, 0.6, 1.0, 0.0],
    [0.40, 0.50, 0.50, 0.7, 1.0, 0.0],
    [0.40, 0.50, 0.60, 0.5, 0.0, 0.0],
    [0.60, 0.60, 0.75, 0.8, 1.0, 1.0],
    [0.70, 0.70, 1.00, 0.9, 1.0, 0.0],
    [0.00, 0.00, 0.00, 0.0, 0.0, 0.0],
    [0.00, 0.00, 0.00, 0.0, 0.0, 0.0],
], dtype=np.float32)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#1 Shape test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_returns_expected_shape():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    assert result.shape == strokes.shape

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2 Formate test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_returns_float32():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    assert result.dtype == np.float32

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3 NaN\Inf test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_returns_only_finite_values():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    assert np.all(np.isfinite(result))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#4 Coordinate test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_keeps_coordinates_in_valid_range():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    x = result[:, 0]
    y = result[:, 1]

    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)
    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#5 Time and pressure test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_keeps_time_and_pressure_in_valid_range():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    t = result[:, 2]
    pressure = result[:, 3]

    assert np.all(t >= 0.0)
    assert np.all(t <= 1.0)
    assert np.all(pressure >= 0.0)
    assert np.all(pressure <= 1.0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#6 Binary features test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_keeps_binary_features_binary():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    pen_down = result[:, 4]
    stroke_start = result[:, 5]

    assert np.all(
        np.isin(pen_down, [0.0, 1.0])
    )
    assert np.all(
        np.isin(stroke_start, [0.0, 1.0])
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#7 Semantic features test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_preserves_semantic_features():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    assert np.array_equal(
        result[:,4],
        strokes[:,4]
    )
    assert np.array_equal(
        result[:, 5],
        strokes[:, 5]
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#8 Padding test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_preserves_padding():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    assert np.allclose(
        result[-2:],
        strokes[-2:]
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#9 Identity transform test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_identity_transform():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 0.0,
        scale = 1.00,
        dx_pixels = 0,
        dy_pixels = 0,
        jitter_std = 0.0,
    )

    assert np.allclose(
        result,
        strokes
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#10 Drawing coordinates changing test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_augment_strokes_changes_drawing_coordinates():

    input_data = strokes
    result = augment_strokes(
        strokes = input_data,
        angle = 10.0,
        scale = 1.05,
        dx_pixels = 2,
        dy_pixels = -2,
        jitter_std = 0.0,
    )

    drawing_mask = strokes[:, 4] >= 0.5

    assert not np.allclose(
        result[drawing_mask, :2],
        strokes[drawing_mask, :2]
    )


def test_augment_training_data_logs_copy_progress(caplog):
    images = np.zeros((1, 64, 64, 1), dtype=np.float32)
    strokes = np.zeros((1, 100, 6), dtype=np.float32)
    labels = np.array([0])

    with caplog.at_level(
        logging.INFO,
        logger="handwriting.training.augmentation",
    ):
        augment_training_data(
            X_train_images=images,
            X_train_strokes=strokes,
            y_train=labels,
            copies_per_sample=1,
        )

    assert "Creating augmented copy 1/1" in caplog.text
