import numpy as np

from handwriting.core.types import FloatArray


def extract_stroke_start(
        pen_down_values: FloatArray,
) -> FloatArray:
    stroke_start_values = np.zeros_like(pen_down_values)
    previous_pen_down = 0.0

    # Detect the beginning of each stroke
    # pen_down:    [1, 1, 1, 0, 1, 1]
    # stroke_start:[1, 0, 0, 0, 1, 0]
    for i, pen_down in enumerate(pen_down_values):
        if pen_down == 1.0 and previous_pen_down == 0.0:
            stroke_start_values[i] = 1.0
        previous_pen_down = pen_down

    return stroke_start_values