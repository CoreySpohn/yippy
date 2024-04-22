"""Logging module for yippy."""

import logging

# Create a logger for the library
logger = logging.getLogger("yippy")
logger.setLevel(logging.INFO)  # Default level

# Setup default handler
handler = logging.StreamHandler()
file_fmt = (
    "[yippy] %(levelname)s %(asctime)s "
    "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
)
formatter = logging.Formatter(file_fmt)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
