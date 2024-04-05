"""Logging module for yippy."""

import logging


def setup_logger(shell_level="INFO", file_level="DEBUG", disable_shell_logging=False):
    """Set up the logger."""
    # Map string level names to logging levels
    level_mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    logger = logging.getLogger(__name__)

    logger.handlers = []  # Clear existing handlers

    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

    # File Handler
    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(level_mapping.get(file_level.upper(), logging.DEBUG))
    file_fmt = (
        "[yippy] %(levelname)s %(asctime)s "
        "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    file_formatter = logging.Formatter(file_fmt)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Shell Handler
    if not disable_shell_logging:
        shell_handler = logging.StreamHandler()
        shell_handler.setLevel(level_mapping.get(shell_level.upper(), logging.INFO))
        shell_fmt = "yippy %(levelname)s [%(asctime)s] %(message)s"
        shell_formatter = logging.Formatter(shell_fmt)
        shell_handler.setFormatter(shell_formatter)
        logger.addHandler(shell_handler)

    logger.propagate = False
    return logger


# Initialize with default settings
logger = setup_logger()
