import logging

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "exalsius": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
    },
}


def get_logger(logger_name: str = "exalsius") -> logging.Logger:
    return logging.getLogger(logger_name)
