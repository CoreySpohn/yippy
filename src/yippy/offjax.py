"""Module for handling off-axis PSFs using JAX."""

from pathlib import Path

import jax.numpy as jnp
from astropy.units import Quantity
from jax import device_put, jit, vmap

from .jax_funcs import (
    convert_xy_1D,
    convert_xy_2DQ,
    create_avg_psf_1D,
    create_avg_psf_2DQ,
    x_basic_shift,
    x_symmetric_shift,
    y_basic_shift,
    y_symmetric_shift,
)
from .offax import OffAx


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
        reshaped_psfs (jnp.ndarray):
            The PSF data, cast as a JAX array.
    """

    def __init__(
        self,
        yip_dir: Path,
        offax_data_file: str,
        offax_offsets_file: str,
        pixel_scale: Quantity,
        x_symmetric: bool,
        y_symmetric: bool,
    ) -> None:
        """Initializes the OffJAX class by casting YIP data to JAX arrays."""
        super().__init__(
            yip_dir,
            offax_data_file,
            offax_offsets_file,
            pixel_scale,
            x_symmetric,
            y_symmetric,
        )

        # Convert the PSF data to JAX arrays
        self.reshaped_psfs = device_put(jnp.array(self.reshaped_psfs))

        self.x_offsets = device_put(jnp.array(self.x_offsets))
        self.y_offsets = device_put(jnp.array(self.y_offsets))
        # Precompute and store coordinate grids based on PSF shape
        height, width = (self.reshaped_psfs.shape[2], self.reshaped_psfs.shape[3])

        self.x_grid, self.y_grid = jnp.meshgrid(
            jnp.arange(width), jnp.arange(height), indexing="xy"
        )
        self.x_grid = device_put(self.x_grid)
        self.y_grid = device_put(self.y_grid)

        if self.type == "1d":
            create_avg_psf = create_avg_psf_1D
            convert_xy = convert_xy_1D
        elif self.type == "2dq":
            create_avg_psf = create_avg_psf_2DQ
            convert_xy = convert_xy_2DQ

        if self.x_symmetric:
            x_shift = x_symmetric_shift
        else:
            x_shift = x_basic_shift
        if self.y_symmetric:
            y_shift = y_symmetric_shift
        else:
            y_shift = y_basic_shift

        def create_psf(x, y):
            psf = create_avg_psf(
                x,
                y,
                self.pixel_scale.value,
                self.x_offsets,
                self.y_offsets,
                self.x_grid,
                self.y_grid,
                self.reshaped_psfs,
            )
            _x, _y = convert_xy(x, y)
            psf = x_shift(x, _x, psf, self.pixel_scale.value)
            psf = y_shift(y, _y, psf, self.pixel_scale.value)
            return psf

        self.create_psf = jit(create_psf)
        self.create_psfs = jit(vmap(self.create_psf, in_axes=(0, 0)))
