"""This module handles the sky transmission map."""

from pathlib import Path

import astropy.io.fits as pyfits


class SkyTrans:
    """Simple class that holds the sky transmission map data.

    This is essentially a placeholder class until I have a better idea of how
    the sky transmission map will be used in the pipeline.

    Attributes:
        data:
            Sky transmission map data.
    """

    def __init__(self, yip_dir: Path, sky_trans_file: str) -> None:
        """Initializes the SkyTrans class by loading the sky transmission map.

        Args:
            yip_dir (Path):
                Path to the directory containing the sky transmission map.
            sky_trans_file (str):
                Name of the file containing the sky transmission map.
        """
        # Load the sky transmission file
        self.data = pyfits.getdata(Path(yip_dir, sky_trans_file), 0)

    def __call__(self):
        """Returns the sky transmission map data."""
        return self.data
