import numpy as np

from handwriting.preprocessing.features import extract_stroke_start


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