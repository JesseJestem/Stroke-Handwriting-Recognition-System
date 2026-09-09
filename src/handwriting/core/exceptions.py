class HandwritingError(Exception):
    """Base exception for handwriting application errors."""


class StrokeDataError(HandwritingError):
    """Raised for errors related to stroke data."""