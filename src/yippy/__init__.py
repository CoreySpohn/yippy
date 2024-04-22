"""yippy allows for a coronagraph object to be created from a yield input package."""

__all__ = ["convert_to_lod", "Coronagraph"]

from .coronagraph import Coronagraph
from .util import convert_to_lod
