"""yippy allows for a coronagraph object to be created from a yield input package."""

__all__ = ["Coronagraph", "setup_logger", "logger"]

from .coronagraph import Coronagraph
from .logger import setup_logger
