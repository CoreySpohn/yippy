"""Module for handling off-axis PSFs using JAX."""

from pathlib import Path

import jax.numpy as jnp
from astropy.units import Quantity
from jax import device_put, jit, vmap

from .offax import OffAx
from .util import (
    create_psf_1D_no_symmetry,
    create_psf_1D_x_symmetry,
    create_psf_1D_xy_symmetry,
    create_psf_1D_y_symmetry,
    create_psf_2DQ_no_symmetry,
    create_psf_2DQ_x_symmetry,
    create_psf_2DQ_xy_symmetry,
    create_psf_2DQ_y_symmetry,
)


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
            # Determine which PSF creation function to use based on symmetry flags
            if self.x_symmetric and self.y_symmetric:
                create_fn = create_psf_1D_xy_symmetry
            elif self.x_symmetric and not self.y_symmetric:
                create_fn = create_psf_1D_x_symmetry
            elif not self.x_symmetric and self.y_symmetric:
                create_fn = create_psf_1D_y_symmetry
            else:
                create_fn = create_psf_1D_no_symmetry
        elif self.type == "2dq":
            if self.x_symmetric and self.y_symmetric:
                create_fn = create_psf_2DQ_xy_symmetry
            elif self.x_symmetric and not self.y_symmetric:
                create_fn = create_psf_2DQ_x_symmetry
            elif not self.x_symmetric and self.y_symmetric:
                create_fn = create_psf_2DQ_y_symmetry
            else:
                create_fn = create_psf_2DQ_no_symmetry

        # Partially apply the necessary arguments except x and y
        def _create_psf(x, y):
            return create_fn(
                x,
                y,
                pixel_scale=self.pixel_scale.value,
                x_offsets=self.x_offsets,
                y_offsets=self.y_offsets,
                x_grid=self.x_grid,
                y_grid=self.y_grid,
                reshaped_psfs=self.reshaped_psfs,
            )

        self.create_psf = jit(_create_psf)

        @jit
        def create_psfs(x, y):
            return jit(vmap(_create_psf)(x, y))

        # create_fn_partial = partial(
        #     create_fn,
        #     pixel_scale=self.pixel_scale.value,
        #     x_offsets=self.x_offsets,
        #     y_offsets=self.y_offsets,
        #     x_grid=self.x_grid,
        #     y_grid=self.y_grid,
        #     reshaped_psfs=self.reshaped_psfs,
        #     static_argnums=(1, 2, 3, 4, 5, 6),
        # )

        # JIT-compile the scalar create_psf function
        # self.create_psf = jit(create_fn_partial)

        # JIT-compile the batched create_psf function using vmap
        # self.create_psfs = jit(vmap(create_fn_partial))
        # self.create_psfs = pmap(jit(create_fn_partial))
