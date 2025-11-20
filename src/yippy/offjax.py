"""Module for handling off-axis PSFs using JAX."""

from functools import partial
from pathlib import Path

import jax.numpy as jnp
from astropy.units import Quantity
from jax import device_put, jit, vmap
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from lod_unit import lod

from .jax_funcs import synthesize_psf_separable
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

        self.x_offsets = device_put(jnp.array(self.x_offsets))
        self.y_offsets = device_put(jnp.array(self.y_offsets))
        n_pixels = self.reshaped_psfs.shape[-1]
        n_pad = int(1.5 * n_pixels)
        n_fft = n_pixels + 2 * n_pad
        n_pad = int(1.5 * n_pixels)
        if (n_pixels + 2 * n_pad) % 2 != 0:
            n_pad += 1

        n_fft = n_pixels + 2 * n_pad  # Guaranteed even

        # Create 1D Frequency Vectors
        # kx: Real-to-Complex frequencies (0 to 0.5)
        self.kx = device_put(jnp.fft.rfftfreq(n_fft))
        # ky: Standard frequencies (0 to 0.5, -0.5 to -1/N)
        self.ky = device_put(jnp.fft.fftfreq(n_fft))

        # Calculate max_offset for bounds check
        max_offset = -1.0
        if self.type == "1d" and hasattr(self, "max_offset_in_image"):
            max_offset = self.max_offset_in_image.to(lod).value

        # Bind the function
        self.create_psf = partial(
            synthesize_psf_separable,
            pixel_scale=self.pixel_scale.value,
            reshaped_psfs=self.reshaped_psfs,
            x_offsets=self.x_offsets,
            y_offsets=self.y_offsets,
            kx=self.kx,
            ky=self.ky,
            n_pad=n_pad,
            x_symmetric=self.x_symmetric,
            y_symmetric=self.y_symmetric,
            input_type=self.type,
            max_offset=max_offset,
        )

        self.create_psfs = vmap(self.create_psf, in_axes=(0, 0))
        self.create_psfs_j = jit(self.create_psfs)

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
            mesh_utils.create_device_mesh([D]),
            axis_names=("i",),
        )

        # Distribute computation across devices. Each device gets N/D items.
        # in_specs=P('i') will shard the first dimension across the 'i' axis.
        psfs_padded = shard_map(
            self.create_psfs_j,
            mesh=mesh,
            in_specs=P("i"),
            out_specs=P("i"),
            check_rep=False,
        )(x_vals_padded, y_vals_padded)

        # Truncate the padded results if we added any padding
        if padding_needed > 0:
            psfs = psfs_padded[:N]
        else:
            psfs = psfs_padded

        return psfs
