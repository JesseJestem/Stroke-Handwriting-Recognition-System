import pytest

from handwriting.core.exceptions import HandwritingError, StrokeDataError


def test_stroke_data_error_can_be_caught_as_handwriting_error():
    with pytest.raises(HandwritingError):
        raise StrokeDataError("invalid stroke data")