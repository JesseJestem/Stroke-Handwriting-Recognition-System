import json

import numpy as np
import pytest

from handwriting.core.exceptions import StrokeDataError
from handwriting.preprocessing.strokes import (
    distribute_points_between_strokes,
    load_stroke_json,
    normalize_coordinates,
    normalize_strokes,
    preprocess_stroke_data,
    preprocess_strokes,
    resample_single_stroke,
    resample_strokes,
    split_strokes,
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#TEST SAMPLES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = {
    "strokes":[
        {"x": 100, "y": 200, "t": 0, "pressure": 0.5, "pen_down": True},
        {"x": 120, "y": 180, "t": 10, "pressure": 0.6, "pen_down": True},
        {"x": 140, "y": 160, "t": 20, "pressure": 0.7, "pen_down": True},
    ]
}

data_stroke_start = {
    "strokes": [
        {"x": 0, "y": 0, "t": 0, "pen_down": True},
        {"x": 1, "y": 1, "t": 1, "pen_down": True},
        {"x": 2, "y": 2, "t": 2, "pen_down": True},

        {"x": 3, "y": 3, "t": 3, "pen_down": False},

        {"x": 4, "y": 4, "t": 4, "pen_down": True},
        {"x": 5, "y": 5, "t": 5, "pen_down": True},
    ]
}

data_pressure = {
    "strokes": [
        {"x": 0, "y": 0, "t": 0, "pressure": -0.5},
        {"x": 1, "y": 1, "t": 1},
        {"x": 2, "y": 2, "t": 2, "pressure": 2.0},
    ]
}

data_empty = {
    "strokes": []
}

strokes = np.array([
    [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
    [0.1, 0.1, 0.1, 0.5, 1.0, 0.0],

    [0.2, 0.2, 0.2, 0.5, 0.0, 0.0],

    [0.3, 0.3, 0.3, 0.5, 1.0, 1.0],
    [0.4, 0.4, 0.4, 0.5, 1.0, 0.0],
], dtype=np.float32)

one_stroke = np.array([
    [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.5, 1.0, 0.0],
], dtype=np.float32)

single_stroke = np.array([
    [0.4, 0.7, 0.2, 0.5, 1.0, 1.0]
], dtype=np.float32)

segment_1 = np.array([
    [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
], dtype=np.float32)

segment_2 = np.array([
    [1.0, 1.0, 1.0, 0.5, 1.0, 1.0]
], dtype=np.float32)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#TESTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_normalize_strokes_returns_expected_shape():

    #Arrange
    input_data = data
    #Act
    result = normalize_strokes(input_data)
    #Assert
    assert result.shape == (3, 6)


def test_normalize_strokes_returns_float32():

    input_data = data
    result = normalize_strokes(input_data)

    assert result.dtype == np.float32


def test_normalize_strokes_normalizes_coordinates():

    input_data = data
    result = normalize_strokes(input_data)

    x = result[:, 0]
    y = result[:, 1]

    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)
    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)


def test_normalize_strokes_normalizes_time():

    input_data = data
    result = normalize_strokes(input_data)
    t = result[:, 2]

    assert np.allclose(
        t,
        [0.0, 0.5, 1.0]
    )


def test_normalize_strokes_detects_stroke_start():

    input_data = data_stroke_start
    result = normalize_strokes(input_data)
    stroke_start = result[:, 5]

    assert np.array_equal(
        stroke_start,
        [1, 0, 0, 0, 1, 0]
    )


def test_normalize_strokes_clips_and_defaults_pressure():
    input_data = data_pressure
    result = normalize_strokes(input_data)
    pressure = result[:, 3]

    assert np.allclose(
        pressure,
        [0.0, 0.5, 1.0]
    )


def test_normalize_strokes_handles_empty_input():

    input_data = data_empty
    result = normalize_strokes(input_data)

    assert result.shape == (0, 6)
    assert result.dtype == np.float32


def test_resample_strokes_returns_requested_number_of_points():

    normalized = normalize_strokes(data)
    result = resample_strokes(normalized, max_points=50)

    assert result.shape == (50, 6)


def test_split_strokes_separates_segments():

    segments = split_strokes(strokes)

    assert len(segments) == 2
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_resample_single_stroke_interpolates_points():

    result = resample_single_stroke(
        one_stroke,
        target_points=5,
    )
    x = result[:, 0]
    y = result[:, 1]
    pen_down = result[:, 4]
    stroke_start = result[:, 5]

    assert result.shape == (5, 6)
    assert np.allclose(
        x,
        [0.00, 0.25, 0.50, 0.75, 1.00]
    )
    assert np.allclose(
        y,
        [0.00, 0.25, 0.50, 0.75, 1.00]
    )
    assert np.array_equal(
        pen_down,
        [1, 1, 1, 1, 1]
    )
    assert np.array_equal(
        stroke_start,
        [1, 0, 0, 0, 0]
    )


def test_resample_single_stroke_repeats_single_point():

    result = resample_single_stroke(
        single_stroke,
        target_points=5,
    )
    x = result[:, 0]
    y = result[:, 1]
    pen_down = result[:, 4]
    stroke_start = result[:, 5]

    assert result.shape == (5, 6)
    assert np.allclose(
        x,
        [0.4, 0.4, 0.4, 0.4, 0.4]
    )
    assert np.allclose(
        y,
        [0.7, 0.7, 0.7, 0.7, 0.7]
    )
    assert np.array_equal(
        pen_down,
        [1, 1, 1, 1, 1]
    )
    assert np.array_equal(
        stroke_start,
        [1, 0, 0, 0, 0]
    )


def test_distribute_points_handles_zero_remaining_points():

    segments = [segment_1, segment_2]
    points_per_segment, separator_points = distribute_points_between_strokes(
        segments=segments,
        max_points=4,
        separator_points_per_gap=2,
    )

    assert np.array_equal(
        points_per_segment,
        [1, 1]
    )
    assert separator_points == 2


def test_load_stroke_json_raises_stroke_data_error_for_missing_file(tmp_path):
    missing_file = tmp_path / 'missing_file.json'

    with pytest.raises(StrokeDataError):
        load_stroke_json(missing_file)


def test_load_stroke_json_raises_stroke_data_error_for_invalid_json(tmp_path):
    invalid_file = tmp_path / 'invalid_file.json'

    invalid_file.write_text(
        "invalid_file.json",
        encoding="utf-8",
    )

    with pytest.raises(StrokeDataError):
        load_stroke_json(invalid_file)


def test_load_stroke_json_returns_stroke_data(tmp_path):
    stroke_file = tmp_path / "stroke.json"

    stroke_file.write_text(
        '{"strokes": [{"x": 1, "y": 2, "t": 3}]}',
        encoding="utf-8",
    )

    result = load_stroke_json(stroke_file)
    assert result["strokes"][0]["x"] == 1


def test_preprocess_stroke_data_returns_expected_shape():
    result = preprocess_stroke_data(
        data,
        max_points=50,
    )

    assert result.shape == (50, 6)
    assert result.dtype == np.float32


def test_preprocess_strokes_matches_preprocess_stroke_data(tmp_path):
    stroke_file = tmp_path / "stroke.json"

    stroke_file.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    from_file = preprocess_strokes(
        stroke_file,
        max_points=50,
    )

    from_data = preprocess_stroke_data(
        data,
        max_points=50,
    )

    np.testing.assert_allclose(from_file, from_data)


def test_normalize_coordinates_preserves_aspect_ratio_and_centers():
    x_values = np.array(
        [0.0, 10.0],
        dtype=np.float32,
    )
    y_values = np.array(
        [0.0, 20.0],
        dtype=np.float32,
    )

    x_norm, y_norm = normalize_coordinates(
        x_values,
        y_values,
    )

    np.testing.assert_allclose(
        x_norm,
        np.array([0.25, 0.75], dtype=np.float32),
    )
    np.testing.assert_allclose(
        y_norm,
        np.array([0.0, 1.0], dtype=np.float32),
    )


def test_normalize_coordinates_returns_zeros_for_degenerate_input():
    x_values = np.array(
        [5.0, 5.0],
        dtype=np.float32,
    )

    y_values = np.array(
        [10.0, 10.0],
        dtype=np.float32,
    )

    x_norm, y_norm = normalize_coordinates(
        x_values,
        y_values,
    )

    np.testing.assert_array_equal(
        x_norm,
        np.zeros((2,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        y_norm,
        np.zeros((2,), dtype=np.float32),
    )

def test_normalize_coordinates_centers_shorter_height():
    x_values = np.array(
        [0.0, 20.0],
        dtype=np.float32,
    )
    y_values = np.array(
        [0.0, 10.0],
        dtype=np.float32,
    )

    x_norm, y_norm = normalize_coordinates(
        x_values,
        y_values,
    )

    np.testing.assert_allclose(
        x_norm,
        np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        y_norm,
        np.array([0.25, 0.75], dtype=np.float32),
    )