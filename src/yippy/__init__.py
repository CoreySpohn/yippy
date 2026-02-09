"""yippy allows for a coronagraph object to be created from a yield input package."""

__all__ = [
    "__version__",
    "Coronagraph",
    "EqxCoronagraph",
    "logger",
]

from ._version import __version__
from .coronagraph import Coronagraph
from .eqx_coronagraph import EqxCoronagraph
from .logger import logger

