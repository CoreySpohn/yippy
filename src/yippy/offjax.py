"""Module for handling off-axis PSFs using JAX."""

from itertools import product
from pathlib import Path

import astropy.units as u
import jax.numpy as np
import xarray as xr
from astropy.units import Quantity
from jax import device_put, jit
from tqdm import tqdm

from .logger import logger
from .offax import OffAx
from .util import create_shift_mask_jax, fft_rotate_jax, fft_shift_jax


class OffJAX(OffAx):
    """Class for handling off-axis PSFs using JAX.

    This class inherits from OffAx and uses JAX for optimized computation.

    Attributes:
        pixel_scale (Quantity):
            Pixel scale of the PSF data in lambda/D.
        center_x (Quantity):
            Central x position in the PSF data.
        center_y (Quantity):
            Central y position in the PSF data.
        reshaped_psfs (np.ndarray):
            The PSF data, cast as a JAX array.
    """

    def __init__(
        self,
        yip_dir: Path,
        offax_data_file: str,
        offax_offsets_file: str,
        pixel_scale: Quantity,
    ) -> None:
        """Initializes the OffJAX class by casting YIP data to JAX arrays."""
        super().__init__(yip_dir, offax_data_file, offax_offsets_file, pixel_scale)

        # Convert the PSF data to JAX arrays
        self.reshaped_psfs = device_put(np.array(self.reshaped_psfs))
        self.x_offsets = device_put(np.array(self.x_offsets))
        self.y_offsets = device_put(np.array(self.y_offsets))
        self.fft_shift = jit(fft_shift_jax)
        self.fft_rotate = jit(fft_rotate_jax)

        # Create the fft_rotate and fft_shift functions

    def create_psf(self, x: float, y: float):
        """Creates and returns the PSF at the specified off-axis position using JAX."""
        # The core logic is similar to OffAx, but uses JAX operations
        rot, flip_lr, flip_ud = 0, False, False

        if x in self.x_offsets and y in self.y_offsets:
            x_ind = np.searchsorted(self.x_offsets, x)
            y_ind = np.searchsorted(self.y_offsets, y)
            return self.reshaped_psfs[x_ind, y_ind]

        if self.type == "1d":
            sep = np.sqrt(x**2 + y**2)
            rot = np.rad2deg(np.arctan2(y, x))
            _x, _y = sep, 0
        elif self.type == "2dq":
            flip_lr, flip_ud = x < 0, y < 0
            _x, _y = abs(x), abs(y)
        else:
            _x, _y = x, y

        _x_ind = np.searchsorted(self.x_offsets, _x)
        _y_ind = np.searchsorted(self.y_offsets, _y)
        if _x in self.x_offsets and _y in self.y_offsets:
            psf = self.reshaped_psfs[_x_ind, _y_ind]
            if flip_lr:
                psf = np.fliplr(psf)
            if flip_ud:
                psf = np.flipud(psf)
            if rot != 0:
                psf = self.fft_rotate(psf, rot)
            return psf

        x_inds = np.array([_x_ind - 1, _x_ind])
        y_inds = np.array([_y_ind - 1, _y_ind])

        x_vals, y_vals = self.x_offsets[x_inds], self.y_offsets[y_inds]
        near_inds = np.array(np.meshgrid(x_inds, y_inds)).T.reshape(-1, 2)
        near_offsets = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)
        near_psfs = self.reshaped_psfs[near_inds[:, 0], near_inds[:, 1]]

        near_shifts = (np.array([_x, _y]) - near_offsets) / self.pixel_scale.value
        near_diffs = np.linalg.norm(near_shifts, axis=1)
        sigma = 1.0
        weights = np.exp(-(near_diffs**2) / (2 * sigma**2))
        weights /= weights.sum()

        psf = np.zeros_like(near_psfs[0])
        weight_array = np.zeros_like(psf)

        for i, near_psf in enumerate(near_psfs):
            shifted_psf = self.fft_shift(near_psf, near_shifts[i][0], near_shifts[i][1])
            weight_mask = create_shift_mask_jax(
                near_psf, near_shifts[i][0], near_shifts[i][1], weights[i]
            )
            psf += weight_mask * shifted_psf
            weight_array += weight_mask

        psf /= weight_array

        if flip_lr:
            psf = np.fliplr(psf)
        if flip_ud:
            psf = np.flipud(psf)
        if rot != 0:
            psf = self.fft_rotate(psf, rot)

        return psf

    def create_offax_datacube(self):
        """Load the disk image from a file or generate it if it doesn't exist."""
        # Load data cube of spatially dependent PSFs.
        path = self.yip_path / "offax_datacube.nc"

        # coords = {
        #     "x psf offset (pix)": np.arange(self.psf_shape[0]),
        #     "y psf offset (pix)": np.arange(self.psf_shape[1]),
        #     "x (pix)": np.arange(self.psf_shape[0]),
        #     "y (pix)": np.arange(self.psf_shape[1]),
        # }
        if path.exists():
            logger.info("Loading data cube of spatially dependent PSFs, please hold...")
            psfs_xr = xr.open_dataarray(path)
        else:
            logger.info(
                "Calculating data cube of spatially dependent PSFs, please hold..."
            )
            # Compute pixel grid.
            # Compute pixel grid contrast.
            psfs_shape = (*self.psf_shape, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            pixel_lod = (
                (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                * u.pixel
                * self.pixel_scale
            ).value

            # x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")
            npsfs = np.prod(self.psf_shape)
            pb = tqdm(total=npsfs, desc="Computing datacube of PSFs at every pixel")

            # Note: intention is that i value maps to x offset and j value maps
            # to y offset

            pix_inds = np.arange(self.npixels)

            for (i, x), (j, y) in product(
                zip(pix_inds, pixel_lod), zip(pix_inds, pixel_lod)
            ):
                psfs[i, j] = self.offax(x, y)
                pb.update(1)
        return psfs_xr
