import json
import logging
from datetime import UTC, datetime

from handwriting.core.request_context import get_request_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id is not None:
            log_data["request_id"] = request_id

        if record.exc_info is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("handwriting")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False