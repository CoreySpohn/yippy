"""Module for handling FITS header data."""

import re
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
from astropy.io import fits
from lod_unit import lod

from .logger import logger


@dataclass(frozen=True)
class HeaderData:
    """A dataclass for storing the header data with units attached.

    I think this will be a useful system, however this has not been tested
    considerably. This is just a suggestion for a more robust way to keep the
    FITS header data attached to the coronagraph. Right now, the header data is
    loaded by Coronagraph from the stellar_intens file. This may make more
    sense if Offax, StellarIntens, and SkyTrans each load their own header.
    """

    simple: bool
    bitpix: int
    naxis: int
    naxis1: int
    naxis2: int
    naxis3: int
    design: str
    diameter: Optional[u.Quantity] = None
    pixscale: Optional[u.Quantity] = None
    lambda0: Optional[u.Quantity] = None
    minlam: Optional[u.Quantity] = None
    maxlam: Optional[u.Quantity] = None
    xcenter: Optional[float] = None
    ycenter: Optional[float] = None
    obscured: Optional[float] = None
    jitter: Optional[u.Quantity] = None
    n_lam: Optional[int] = None
    n_star: Optional[int] = None
    zernike: Optional[str] = None
    wfe: Optional[u.Quantity] = None

    @staticmethod
    def extract_unit(comment: str, default_unit: u.Unit, key: str) -> u.Unit:
        """Extract a unit from the comment string of a FITS header entry.

        This attempts to read the FITS header comment and search for an astropy
        Unit. This seems fairly robust, but it may not catch all cases in which
        case a warning is raised. It is also possible that the unit is not
        explicitly stated in the comment, in which case a default unit is
        returned.

        Args:
            comment (str):
                The comment string associated with a FITS header key.
            default_unit (u.Unit):
                The default unit to return if no unit is found in the comment.
            key (str):
                The key of the header entry to extract the unit from.

        Returns:
            u.Unit:
                The extracted astropy unit or the default unit if no
                recognizable unit is found.

        Raises:
            Warning:
                Logs a warning if no unit could be extracted and the default
                unit is used.
        """
        # Define a regex pattern for commonly expected units
        unit_patterns = {
            "nm": u.nm,
            "micron": u.micron,
            "microns": u.micron,
            "um": u.um,
            "mm": u.mm,
            "cm": u.cm,
            "meter": u.m,
            "m": u.m,
            "arcsec": u.arcsec,
            "arcmin": u.arcmin,
            "deg": u.deg,
            "mas": u.mas,
            "pm": u.pm,
        }

        # Match for patterns that look like compound units, e.g., "mas/pixel"
        compound_unit_pattern = r"\b(\w+)/(\w+)\b"
        match = re.search(compound_unit_pattern, comment, re.IGNORECASE)
        if match:
            num_unit, den_unit = match.groups()
            num_astropy_unit = unit_patterns.get(num_unit.lower(), None)
            den_astropy_unit = unit_patterns.get(den_unit.lower(), None)
            if num_astropy_unit and den_astropy_unit:
                logger.debug(
                    f"Extracted compound unit {num_astropy_unit}/{den_astropy_unit}"
                    f' for {key} from "{comment}"'
                )
                return num_astropy_unit / den_astropy_unit

        # Search for any of these single units in the comment
        for pattern, unit in unit_patterns.items():
            if re.search(r"\b" + re.escape(pattern) + r"\b", comment, re.IGNORECASE):
                logger.debug(f'Extracted {unit} from "{comment}" for {key}')
                return unit

        logger.warning(
            (
                f"Using default unit for {key}: {default_unit}. "
                f'Could not extract unit from comment: "{comment}"'
            )
        )
        return default_unit

    @staticmethod
    def get_header_value(header: fits.Header, key: str, default_unit: u.Unit):
        """Retrieves a header value by key and converts it to a specified unit.

        Args:
            header (fits.Header):
                The FITS header from which to retrieve the value.
            key (str):
                The key of the header entry to retrieve.
            default_unit (u.Unit):
                The unit to associate with the returned value.

        Returns:
            Optional[u.Quantity]:
                The header value with the specified unit, or None if the key is
                not present.
        """
        if key in header:
            value = float(header[key])
            unit = HeaderData.extract_unit(header.comments[key], default_unit, key)
            return value * unit
        else:
            return None

    @staticmethod
    def from_fits_header(header: fits.Header):
        """Parses a FITS header.

        This parses the FITS header to initialize the HeaderData class,
        checking for any unhandled fields.

        Args:
            header (fits.Header):
                The FITS header to parse.

        Returns:
            HeaderData:
                An initialized HeaderData object populated with values from the
                FITS header.

        Raises:
            Warning:
                Logs a warning for any unhandled fields in the FITS header.
        """
        recognized_keys = set(
            [
                "SIMPLE",
                "BITPIX",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "NAXIS3",
                "DESIGN",
                "D",
                "PIXSCALE",
                "LAMBDA",
                "MINLAM",
                "MAXLAM",
                "XCENTER",
                "YCENTER",
                "OBSCURED",
                "JITTER",
                "N_LAM",
                "N_STAR",
                "ZERNIKE",
                "WFE",
            ]
        )
        header_keys = set(header.keys())

        unhandled_keys = header_keys - recognized_keys
        if unhandled_keys:
            logger.warning(f"Unhandled header fields: {unhandled_keys}")

        return HeaderData(
            simple=header.get("SIMPLE", "F") == "T",
            bitpix=int(header.get("BITPIX", 0)),
            naxis=int(header.get("NAXIS", 0)),
            naxis1=int(header.get("NAXIS1", 0)),
            naxis2=int(header.get("NAXIS2", 0)),
            naxis3=int(header.get("NAXIS3", 0)),
            design=header.get("DESIGN", ""),
            diameter=HeaderData.get_header_value(header, "D", u.meter),
            pixscale=header.get("PIXSCALE") * lod / u.pix,
            lambda0=HeaderData.get_header_value(header, "LAMBDA", u.micron),
            minlam=HeaderData.get_header_value(header, "MINLAM", u.micron),
            maxlam=HeaderData.get_header_value(header, "MAXLAM", u.micron),
            xcenter=header.get("XCENTER", None),
            ycenter=header.get("YCENTER", None),
            obscured=header.get("OBSCURED", None),
            jitter=HeaderData.get_header_value(header, "JITTER", u.mas),
            n_lam=header.get("N_LAM", None),
            n_star=header.get("N_STAR", None),
            zernike=header.get("ZERNIKE", ""),
            wfe=HeaderData.get_header_value(header, "WFE", u.pm),
        )
