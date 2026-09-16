import pytest

from handwriting.core.exceptions import StrokeDataError
from handwriting.data.validation import validate_stroke_data

VALID_DATA = {
    "strokes": [
        {
            "x": 10.0,
            "y": 20.0,
            "t": 0.0,
            "pressure": 0.5,
            "pen_down": True,
        }
    ],
    "label": "a",
    "canvas_width": 400,
    "canvas_height": 400,
}

INVALID_STROKES_DATA = {
    "strokes": "invalid strokes",
    "label": "a",
    "canvas_width": 400,
    "canvas_height": 400,
}

INVALID_POINT_DATA = {
    "strokes": [
        "invalid point"
    ],
}


DATA_ONLY_X_Y_T = {
        "strokes": [
            {
                "x": 10.0,
                "y": 20.0,
                "t": 0.0,
            }
        ],
    }


def test_validate_stroke_data_returns_valid_data():
    result = validate_stroke_data(
        data=VALID_DATA,
    )

    assert result == VALID_DATA

def test_validate_stroke_data_rejects_non_dictionary():
    with pytest.raises(StrokeDataError):
        validate_stroke_data(data=[])

def test_validate_stroke_data_rejects_invalid_strokes():
    with pytest.raises(StrokeDataError):
        validate_stroke_data(
            data=INVALID_STROKES_DATA,
        )


def test_validate_stroke_data_rejects_missing_strokes():
    with pytest.raises(StrokeDataError):
        validate_stroke_data(data={})


def test_validate_stroke_data_rejects_non_dictionary_point():
    with pytest.raises(StrokeDataError):
        validate_stroke_data(
            data=INVALID_POINT_DATA,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "x",
        "y",
        "t",
    ],
)
def test_validate_stroke_data_rejects_missing_required_point_field(missing_field):
    point = {
        "x": 10.0,
        "y": 20.0,
        "t": 0.0,
    }

    del point[missing_field]

    invalid_data = {
        "strokes": [point],
    }

    with pytest.raises(StrokeDataError):
        validate_stroke_data(
            data=invalid_data,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("x", "wrong"),
        ("x", True),
        ("y", None),
        ("y", False),
        ("t", []),
        ("t", True),
    ],
)
def test_validate_stroke_data_rejects_invalid_required_field_type(field, invalid_value):
    point = {
        "x": 10.0,
        "y": 20.0,
        "t": 0.0,
    }

    point[field] = invalid_value

    invalid_data = {
        "strokes": [point],
    }

    with pytest.raises(StrokeDataError):
        validate_stroke_data(
            data=invalid_data,
        )


def test_validate_stroke_data_accepts_missing_optional_point_fields():
    result = validate_stroke_data(DATA_ONLY_X_Y_T)

    assert result == DATA_ONLY_X_Y_T


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("pressure", "wrong"),
        ("pressure", True),
        ("pressure", None),
        ("pen_down", 1),
        ("pen_down", "true"),
        ("pen_down", None),
    ],
)
def test_validate_stroke_data_rejects_invalid_optional_point_field_type(
    field,
    invalid_value,
):
    point = {
        "x": 10.0,
        "y": 20.0,
        "t": 0.0,
    }

    point[field] = invalid_value

    invalid_data = {
        "strokes": [point],
    }

    with pytest.raises(StrokeDataError):
        validate_stroke_data(
            data=invalid_data,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("label", 123),
        ("label", None),
        ("canvas_width", 400.5),
        ("canvas_width", True),
        ("canvas_height", "400"),
        ("canvas_height", False),
    ],
)
def test_validate_stroke_data_rejects_invalid_optional_metadata_type(
    field,
    invalid_value,
):
    invalid_data = {
        "strokes": [],
        field: invalid_value,
    }

    with pytest.raises(StrokeDataError):
        validate_stroke_data(
            data=invalid_data,
        )


def test_validate_stroke_data_accepts_integer_numeric_values():
    data = {
        "strokes": [
            {
                "x": 10,
                "y": 20,
                "t": 0,
                "pressure": 1,
            }
        ],
    }

    result = validate_stroke_data(data)

    assert result == data