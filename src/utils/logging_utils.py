import logging

import colorlog
from lightning.pytorch.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_colorlogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger with color support."""
    logger = get_logger(name)

    # define the color log formatter
    fmt = "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
    fmt += "[%(log_color)s%(levelname)s%(reset)s] - %(message)s"
    log_colors = {
        "DEBUG": "purple",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }
    color_formatter = colorlog.ColoredFormatter(fmt=fmt, log_colors=log_colors)

    # configure the console handler with the color log formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)

    # add the handlers to the logger and set the logging level
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger
