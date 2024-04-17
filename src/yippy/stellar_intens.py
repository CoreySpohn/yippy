"""Module for handling stellar intensity data from stellar_intens files."""

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.io.fits as pyfits
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from scipy.interpolate import CubicSpline

from .util import convert_to_lod

if TYPE_CHECKING:
    from logging import Logger


class StellarIntens:
    """Class to handle and interpolate stellar intensity data.

    This class loads stellar intensity data and corresponding stellar diameters,
    providing an interpolation interface to get the stellar intensity map for a given
    stellar diameter.

    Attributes:
        ln_interp (CubicSpline):
            A spline interpolator that operates on the natural logarithm of the
            stellar intensities to ensure that interpolated values are non-negative.

    Args:
        yip_dir (Path):
            Path to the directory containing the yield input package (YIP).
        logger (Logger):
            Logger for logging information and debugging.
        stellar_intens_file (str):
            Filename of the FITS file containing the stellar intensity data.
        stellar_diam_file (str):
            Filename of the FITS file containing the stellar diameters.
    """

    def __init__(
        self,
        yip_dir: Path,
        logger: "Logger",
        stellar_intens_file: str,
        stellar_diam_file: str,
    ) -> None:
        """Initializes StellarIntens class by loading data and creating the interpolant.

        Stellar intensity data and stellar diameters are loaded from FITS files,
        and a natural logarithm-based cubic spline interpolator is set up to
        facilitate interpolation.

        Args:
            yip_dir (Path):
                The directory where the stellar intensity and diameter FITS
                files are located.
            logger (Logger):
                Logger for logging setup and processing steps.
            stellar_intens_file (str):
                The filename of the FITS file containing unitless arrays of stellar
                intensities.
            stellar_diam_file (str):
                The filename of the FITS file containing arrays of stellar diameters in
                lambda/D units.
        """
        # Load on-axis stellar intensity PSF data
        psfs = pyfits.getdata(Path(yip_dir, stellar_intens_file), 0)

        # Load the stellar angular diameters in units of lambda/D
        diams = pyfits.getdata(Path(yip_dir, stellar_diam_file), 0) * lod

        # Interpolate stellar data in logarithmic space to ensure non-negative
        # interpolated values
        self.ln_interp = CubicSpline(diams, np.log(psfs))

    def __call__(self, stellar_diam: Quantity, lam=None, D=None):
        """Returns the stellar intensity map at a specified stellar diameter.

        Stellar intensity is interpolated based on a specified diameter using the
        previously configured cubic spline interpolator. The interpolation considers
        the logarithm of the intensity to ensure that all computed values are positive.

        Args:
            stellar_diam (Quantity):
                The desired stellar diameter for which to retrieve the intensity map, in
                units of lambda/D.
            lam (Quantity, optional):
                Wavelength of observation, required if stellar_diam is in angular units.
            D (Quantity, optional):
                Diameter of the telescope, required if stellar_diam is in angular units.

        Returns:
            NDArray:
                An interpolated 2D map of the stellar intensity at the given
                stellar diameter.
        """
        # Set up conversion from angular units to lambda/D
        if stellar_diam.unit != lod:
            stellar_diam = convert_to_lod(stellar_diam, lam=lam, D=D)

        # Return the exponentiated result of the interpolation to get the
        # actual intensity map
        return np.exp(self.ln_interp(stellar_diam))
