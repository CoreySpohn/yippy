"""Module for handling off-axis PSFs using JAX."""

from pathlib import Path

import jax.numpy as jnp
from astropy.units import Quantity
from jax import device_put, jit, vmap
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from lod_unit import lod

from .jax_funcs import synthesize_psf_idw, synthesize_psf_separable
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
        cpu_cores: int = 4,
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
        # For 2D IDW interpolation
        self.flat_psfs = device_put(jnp.array(self.flat_psfs))
        self.flat_x_offsets = device_put(jnp.array(self.flat_offsets[:, 0]))
        self.flat_y_offsets = device_put(jnp.array(self.flat_offsets[:, 1]))

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
        if self.type == "1d":
            # Optimized based on the assumption that the data is on a 1D grid
            self.psf_handle = self.reshaped_psfs
            self.x_handle = self.x_offsets
            self.y_handle = self.y_offsets

            max_offset = -1.0
            if hasattr(self, "max_offset_in_image"):
                max_offset = self.max_offset_in_image.to(lod).value

            def create_psf_kernel(x, y, psfs, x_off, y_off, kx, ky):
                return synthesize_psf_separable(
                    x,
                    y,
                    pixel_scale=self.pixel_scale.value,
                    reshaped_psfs=psfs,
                    x_offsets=x_off,
                    y_offsets=y_off,
                    kx=kx,
                    ky=ky,
                    n_pad=n_pad,
                    x_symmetric=self.x_symmetric,
                    y_symmetric=self.y_symmetric,
                    input_type=self.type,
                    max_offset=max_offset,
                )

        else:
            # 2D, uses a different interpolation method for irregular grids
            self.psf_handle = self.flat_psfs
            self.x_handle = self.flat_x_offsets
            self.y_handle = self.flat_y_offsets

            def create_psf_kernel(x, y, psfs, x_off, y_off, kx, ky):
                return synthesize_psf_idw(
                    x,
                    y,
                    pixel_scale=self.pixel_scale.value,
                    flat_psfs=psfs,
                    flat_x_offsets=x_off,
                    flat_y_offsets=y_off,
                    kx=kx,
                    ky=ky,
                    n_pad=n_pad,
                    x_symmetric=self.x_symmetric,
                    y_symmetric=self.y_symmetric,
                    input_type=self.type,
                    k_neighbors=4,
                )

        # ---------------------------------------------------------
        # BINDING
        # ---------------------------------------------------------
        self.create_psfs_kernel = vmap(
            create_psf_kernel, in_axes=(0, 0, None, None, None, None, None)
        )
        self.create_psfs_j = jit(self.create_psfs_kernel)
        self.create_psf_kernel_single = jit(create_psf_kernel)

        # ---------------------------------------------------------
        # WRAPPERS
        # ---------------------------------------------------------
        # These now use the handles selected above (Grid vs Cloud)

        def create_psfs_wrapper(x, y):
            return self.create_psfs_j(
                x,
                y,
                self.psf_handle,
                self.x_handle,
                self.y_handle,
                self.kx,
                self.ky,
            )

        self.create_psfs = create_psfs_wrapper

        def create_psf_wrapper(x, y):
            return self.create_psf_kernel_single(
                x,
                y,
                self.psf_handle,
                self.x_handle,
                self.y_handle,
                self.kx,
                self.ky,
            )

        self.create_psf = create_psf_wrapper

    def create_psfs_parallel(self, x_vals, y_vals):
        """Create off-axis PSFs at multiple positions in parallel using shard_map."""
        D = self.cpu_cores
        N = x_vals.shape[0]

        remainder = N % D
        padding_needed = (D - remainder) if remainder != 0 else 0

        if padding_needed > 0:
            x_vals_padded = jnp.pad(x_vals, (0, padding_needed), constant_values=0)
            y_vals_padded = jnp.pad(y_vals, (0, padding_needed), constant_values=0)
        else:
            x_vals_padded = x_vals
            y_vals_padded = y_vals

        mesh = Mesh(
            mesh_utils.create_device_mesh([D]),
            axis_names=("i",),
        )

        psfs_padded = shard_map(
            self.create_psfs_j,
            mesh=mesh,
            in_specs=(P("i"), P("i"), P(), P(), P(), P(), P()),
            out_specs=P("i"),
            check_rep=False,
        )(
            x_vals_padded,
            y_vals_padded,
            self.psf_handle,
            self.x_handle,
            self.y_handle,
            self.kx,
            self.ky,
        )

        if padding_needed > 0:
            psfs = psfs_padded[:N]
        else:
            psfs = psfs_padded

        return psfs
