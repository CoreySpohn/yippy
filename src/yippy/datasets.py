"""Example data management for yippy.

This module provides utilities for downloading and accessing example yield
input packages (YIPs) for testing and documentation.  It uses `pooch` to
handle data downloads and caching so that users don't need to ship large
FITS files alongside their code.

Example:
-------
>>> from yippy.datasets import fetch_coronagraph
>>> yip_path = fetch_coronagraph()
>>> from yippy import Coronagraph
>>> coro = Coronagraph(yip_path)
"""

from __future__ import annotations

import pooch
from pooch import Unzip

# ---------------------------------------------------------------------------
# Pooch registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, str] = {
    "coronagraphs.zip": "md5:1537f41c20cb10170537a7d4e89f64b2",
}

# Pooch instance – downloads are cached in the OS-specific user cache dir.
# The base_url points at yippy's ``data/`` directory on GitHub.
PIKACHU = pooch.create(
    path=pooch.os_cache("yippy"),
    base_url="https://github.com/CoreySpohn/yippy/raw/main/data/",
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def fetch_coronagraph(name: str = "eac1_aavc_512") -> str:
    """Download and unpack an example coronagraph YIP.

    The archive contains a ready-to-use yield input package that can be
    passed directly to :class:`yippy.Coronagraph`.

    Args:
        name: Directory name inside the archive.  The default
            ``"eac1_aavc_512"`` is an amplitude apodized vortex coronagraph
            created by Susan Redmond.

    Returns:
        Absolute path to the unpacked YIP directory.

    Example:
        >>> yip_path = fetch_coronagraph()
        >>> from yippy import Coronagraph
        >>> coro = Coronagraph(yip_path)
    """
    PIKACHU.fetch("coronagraphs.zip", processor=Unzip())
    return str(PIKACHU.abspath / "coronagraphs.zip.unzip" / "coronagraphs" / name)
