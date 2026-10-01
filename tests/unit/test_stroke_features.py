import numpy as np

from handwriting.preprocessing.features import (
    extract_coordinate_deltas,
    extract_stroke_start,
)


def test_extract_stroke_start_detects_stroke_beginnings():
    pen_down_values = np.array(
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )

    result = extract_stroke_start(pen_down_values)

    expected = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result, expected)


def test_extract_stroke_start_handles_sequence_starting_with_pen_up():
    pen_down_values = np.array(
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )

    result = extract_stroke_start(pen_down_values)

    expected = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result, expected)


def test_extract_stroke_start_marks_only_first_point_of_continuous_stroke():
    pen_down_values = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    result = extract_stroke_start(pen_down_values)

    expected = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result, expected)


def test_extract_coordinate_deltas_returns_expected_values():
    x_values = np.array(
        [0.0, 0.2, 0.5],
        dtype=np.float32,
    )
    y_values = np.array(
        [0.1, 0.4, 0.4],
        dtype=np.float32,
    )

    dx, dy = extract_coordinate_deltas(
        x_values,
        y_values,
    )

    np.testing.assert_allclose(
        dx,
        np.array([0.0, 0.2, 0.3], dtype=np.float32),
    )
    np.testing.assert_allclose(
        dy,
        np.array([0.0, 0.3, 0.0], dtype=np.float32),
    )


def test_extract_coordinate_negative_deltas_returns_expected_values():
    x_values = np.array(
        [0.5, 0.2, 0.1],
        dtype=np.float32,
    )
    y_values = np.array(
        [0.6, 0.4, 0.7],
        dtype=np.float32,
    )

    dx, dy = extract_coordinate_deltas(
        x_values,
        y_values,
    )

    np.testing.assert_allclose(
        dx,
        np.array([0.0, -0.3, -0.1], dtype=np.float32),
    )
    np.testing.assert_allclose(
        dy,
        np.array([0.0, -0.2, 0.3], dtype=np.float32),
    )


def test_extract_coordinate_deltas_returns_zeros_for_single_point():
    x_values = np.array(
        [0.5],
        dtype=np.float32,
    )
    y_values = np.array(
        [0.7],
        dtype=np.float32,
    )

    dx, dy = extract_coordinate_deltas(
        x_values,
        y_values,
    )

    np.testing.assert_array_equal(
        dx,
        np.array([0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dy,
        np.array([0.0], dtype=np.float32),
    )


def test_extract_coordinate_deltas_handles_empty_arrays():
    x_values = np.array([], dtype=np.float32)
    y_values = np.array([], dtype=np.float32)

    dx, dy = extract_coordinate_deltas(
        x_values,
        y_values,
    )

    np.testing.assert_array_equal(
        dx,
        np.array([], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dy,
        np.array([], dtype=np.float32),
    )