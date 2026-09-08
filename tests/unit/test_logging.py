import json
import logging
import sys

from handwriting.core.logging import JsonFormatter, configure_logging


def test_json_formatter_returns_structured_log():
    record = logging.LogRecord(
        name="handwriting.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatter = JsonFormatter()

    formatted_log = formatter.format(record)
    data = json.loads(formatted_log)

    assert "timestamp" in data
    assert data["level"] == "INFO"
    assert data["logger"] == "handwriting.test"
    assert data["message"] == "Test message"


def test_configure_logging_sets_handwriting_logger():
    configure_logging(level=logging.DEBUG)

    logger = logging.getLogger("handwriting")

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[0].formatter, JsonFormatter)
    assert logger.propagate is False

def test_json_formatter_includes_exception_details():
    formatter = JsonFormatter()

    try:
        raise ValueError("test error")

    except ValueError:
        record = logging.LogRecord(
            name="handwriting.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="Operation failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    data = json.loads(formatter.format(record))

    assert "exception" in data
    assert "ValueError" in data["exception"]
    assert "test error" in data["exception"]