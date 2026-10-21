class HandwritingError(Exception):
    """Base exception for handwriting application errors."""


class StrokeDataError(HandwritingError):
    """Raised for errors related to stroke data."""


class DatasetManifestError(HandwritingError):
    """Raised for errors related to dataset manifests."""