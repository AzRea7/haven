import logging
import json
import sys
import time
from .config import config

class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # attach contextual info if provided
        ctx = getattr(record, "context", None)
        if isinstance(ctx, dict):
            payload.update(ctx)
        return json.dumps(payload)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(config.LOG_LEVEL)
        logger.propagate = False
    return logger
