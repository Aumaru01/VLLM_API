import logging

from logging.handlers import TimedRotatingFileHandler

from __init__ import LOG_DIR, LOG_RETENTION_DAYS, LOG_LEVEL

_LOGGER_NAME = "vllm_api"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rolls over to a new file every midnight; keeps only the last
    # LOG_RETENTION_DAYS files and deletes older ones automatically.
    file_handler = TimedRotatingFileHandler(
        LOG_DIR / f"{_LOGGER_NAME}.log",
        when="midnight",
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = get_logger()
