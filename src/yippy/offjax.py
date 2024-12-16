"""Module for handling off-axis PSFs using JAX."""

from pathlib import Path

import jax.numpy as jnp
from astropy.units import Quantity
from jax import device_put, jit, vmap
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from .jax_funcs import (
    basic_shift_val,
    convert_xy_1D,
    convert_xy_2DQ,
    create_avg_psf_1D,
    create_avg_psf_2DQ,
    create_shift_mask,
    get_pad_info,
    sym_shift_val,
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
        cpu_cores: int = 1,
        platform: str = "cpu",
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
        self.cpu_cores = cpu_cores
        self.platform = platform

        ##############
        # Convert the PSF data to JAX arrays
        ##############
        self.reshaped_psfs = device_put(jnp.array(self.reshaped_psfs))
        n_pixels_orig, n_pad, img_edge, n_pixels_final = get_pad_info(
            self.reshaped_psfs[0, 0], 1.5
        )

        self.x_offsets = device_put(jnp.array(self.x_offsets))
        self.y_offsets = device_put(jnp.array(self.y_offsets))
        # Precompute and store coordinate grids based on PSF shape
        height, width = (self.reshaped_psfs.shape[2], self.reshaped_psfs.shape[3])
        # height, width = (self.padded_psfs.shape[2], self.padded_psfs.shape[3])

        self.x_grid, self.y_grid = jnp.meshgrid(
            jnp.arange(width), jnp.arange(height), indexing="xy"
        )
        self.x_grid = device_put(self.x_grid)
        self.y_grid = device_put(self.y_grid)

        # Create the frequency grids
        ky = jnp.fft.fftfreq(n_pixels_final)
        kx = jnp.fft.fftfreq(n_pixels_final)

        # Precomputed base exponentials
        self.x_phasor = jnp.exp(-2j * jnp.pi * kx)
        self.y_phasor = jnp.exp(-2j * jnp.pi * ky)

        ##############
        # Choose the transformations
        ##############
        if self.type == "1d":
            create_avg_psf = create_avg_psf_1D
            convert_xy = convert_xy_1D
        elif self.type == "2dq":
            create_avg_psf = create_avg_psf_2DQ
            convert_xy = convert_xy_2DQ

        if self.x_symmetric:
            x_shift = x_symmetric_shift
            x_shift_val = sym_shift_val
        else:
            x_shift = x_basic_shift
            x_shift_val = basic_shift_val
        if self.y_symmetric:
            y_shift = y_symmetric_shift
            y_shift_val = sym_shift_val
        else:
            y_shift = y_basic_shift
            y_shift_val = basic_shift_val

        ##############
        # Create the JAX functions
        ##############
        def create_psf(x, y):
            """Create an off-axis PSF at a given position.

            This uses closures to pass in the reshaped PSFs, pixel scale, and
            offsets which allows for JIT compilation that treats them as
            constants.

            Args:
                x (float):
                    x position in lambda/D.
                y (float):
                    y position in lambda/D.

            Returns:
                jnp.ndarray:
                    The off-axis PSF at the given position.
            """
            avg_psf = create_avg_psf(
                x,
                y,
                self.pixel_scale.value,
                self.x_offsets,
                self.y_offsets,
                self.x_grid,
                self.y_grid,
                self.x_phasor,
                self.y_phasor,
                self.reshaped_psfs,
            )
            _x, _y = convert_xy(x, y)
            psf = x_shift(x, _x, avg_psf, self.pixel_scale.value, self.x_phasor)
            psf = y_shift(y, _y, psf, self.pixel_scale.value, self.y_phasor)
            _x_shift = x_shift_val(x, _x, self.pixel_scale.value)
            _y_shift = y_shift_val(y, _y, self.pixel_scale.value)
            mask = create_shift_mask(
                psf, _x_shift, _y_shift, self.x_grid, self.y_grid, 1
            )
            return psf * mask

        self.create_psf = jit(create_psf)
        self.create_psfs = jit(vmap(create_psf, in_axes=(0, 0)))

    def create_psfs_parallel(self, x_vals, y_vals):
        """Create off-axis PSFs at multiple positions in parallel using shard_map.

        If the number of (x, y) pairs doesn't evenly divide the number of devices,
        we pad the inputs and then discard the extra results after processing.

        Args:
            x_vals (jnp.ndarray):
                Array of x-coordinates of shape (N,).
            y_vals (jnp.ndarray):
                Array of y-coordinates of shape (N,).

        Returns:
            jnp.ndarray:
                Array of off-axis PSFs of shape (N, height, width).
        """
        D = self.cpu_cores  # Number of devices
        N = x_vals.shape[0]

        # Determine if we need padding
        remainder = N % D
        padding_needed = (D - remainder) if remainder != 0 else 0

        # Pad inputs so total number of items is divisible by D
        if padding_needed > 0:
            x_vals_padded = jnp.pad(x_vals, (0, padding_needed), constant_values=0)
            y_vals_padded = jnp.pad(y_vals, (0, padding_needed), constant_values=0)
        else:
            x_vals_padded = x_vals
            y_vals_padded = y_vals

        # Set up the device mesh for shard_map
        mesh = Mesh(
            mesh_utils.create_device_mesh(D),
            axis_names=("i",),
        )

        # Distribute computation across devices. Each device gets N/D items.
        # in_specs=P('i') will shard the first dimension across the 'i' axis.
        psfs_padded = shard_map(
            self.create_psfs,
            mesh=mesh,
            in_specs=P("i"),
            out_specs=P("i"),
        )(x_vals_padded, y_vals_padded)

        # Truncate the padded results if we added any padding
        if padding_needed > 0:
            psfs = psfs_padded[:N]
        else:
            psfs = psfs_padded

        return psfs
