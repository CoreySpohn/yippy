"""Utility functions for the yippy package."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lod_unit import lod, lod_eq


def convert_to_lod(
    x: Quantity, center_pix=None, pixel_scale=None, lam=None, D=None, dist=None
) -> Quantity:
    """Convert the x/y position to lambda/D.

    This function has the following assumptions on the x/y values provided:
        - If units are pixels, they follow the 00LL convention. As in the (0,0)
          point is the lower left corner of the image.
        - If the x/y values are in lambda/D, angular, or length units the
            (0,0) point is the center of the image, where the star is
            (hopefully) located.

    Args:
        x (astropy.units.Quantity):
            Position. Can be units of pixel, an angular unit (e.g. arcsec),
            or a length unit (e.g. AU)
        center_pix (astropy.units.Quantity):
            Center of the image in pixels (for the relevant axis)
        pixel_scale (astropy.units.Quantity):
            Pixel scale of in
        lam (astropy.units.Quantity):
            Wavelength of the observation
        D (astropy.units.Quantity):
            Diameter of the telescope
        dist (astropy.units.Quantity):
            Distance to the system
    """
    if x.unit == "pixel":
        assert (
            center_pix is not None
        ), "Center pixel must be provided to convert pixel to lod."
        assert (
            pixel_scale is not None
        ), "Pixel scale must be provided to convert pixel to lod."
        assert pixel_scale.unit == (
            lod / u.pix
        ), f"Pixel scale must be in units of lod/pix, not {pixel_scale.unit}."

        x = x - center_pix
        x = x * pixel_scale
        # Center the x position
    elif x.unit.physical_type == "angle":
        assert lam is not None, (
            f"Wavelength must be provided to convert {x.unit.physical_type}" f" to lod."
        )
        assert D is not None, (
            f"Telescope diameter must be provided to convert {x.unit.physical_type}"
            f" to lod."
        )
        x = x.to(lod, lod_eq(lam, D))
    elif x.unit.physical_type == "length":
        # If the distance to the system is not provided, raise an error
        assert dist is not None, (
            f"Distance to system must be provided to convert {x.unit.physical_type}"
            f" to {lod}."
        )
        x_angular = np.arctan(x.to(u.m).value / dist.to(u.m).value) * u.rad
        x = x_angular.to(lod, lod_eq(lam, D))
    else:
        raise ValueError(f"No conversion implemented for {x.unit.physical_type}")
    return x
