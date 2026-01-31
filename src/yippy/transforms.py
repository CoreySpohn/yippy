"""Image transformation utilities for yippy.

This module contains cubic spline interpolation and resampling functions adapted
from coronagraphoto. The map_coordinates implementation is based on work from
the JAX project (PR #14218 by Louis Desdoigts).

Original JAX License Notice:
Copyright 2019 The JAX Authors.
Licensed under the Apache License, Version 2.0.
"""

import functools
import itertools
import operator
from typing import Callable, Dict, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax._src import api, util
from jax._src.numpy.linalg import inv
from jax._src.typing import Array, ArrayLike
from jax._src.util import safe_zip as zip

# =============================================================================
# Map Coordinates (Cubic Spline Interpolation)
# =============================================================================


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1
    return jnp.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: Array, size: int) -> Array:
    return jnp.floor_divide(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)


_INDEX_FIXERS: Dict[str, Callable[[Array, int], Array]] = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: jnp.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(coordinate: Array) -> List[Tuple[Array, ArrayLike]]:
    index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> List[Tuple[Array, ArrayLike]]:
    lower = jnp.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _cubic_indices_and_weights(coordinate: Array) -> List[Tuple[Array, ArrayLike]]:
    return [(coordinate, jnp.zeros(coordinate.shape))]


def _build_matrix(n: int, diag: float = 4) -> Array:
    A = diag * jnp.eye(n)
    for i in range(n - 1):
        A = A.at[i, i + 1].set(1)
        A = A.at[i + 1, i].set(1)
    return A


def _construct_vector(data: Array, c2: Array, cnp2: Array) -> Array:
    yvec = data[1:-1]
    first = data[1] - c2
    last = data[-2] - cnp2
    yvec = yvec.at[0].set(first)
    yvec = yvec.at[-1].set(last)
    return yvec


def _solve_coefficients(data: Array, A_inv: Array, h=1) -> Array:
    c2 = 1 / 6 * data[0]
    cnp2 = 1 / 6 * data[-1]
    yvec = _construct_vector(data, c2, cnp2)
    cs = jnp.dot(A_inv, yvec)
    c1 = 2 * c2 - cs[0]
    cnp3 = 2 * cnp2 - cs[-1]
    return jnp.concatenate([jnp.array([c1, c2]), cs, jnp.array([cnp2, cnp3])])


def _spline_coefficients(data: Array) -> Array:
    ndim = data.ndim
    for i in range(ndim):
        axis = ndim - i - 1
        A_inv = inv(_build_matrix(data.shape[axis] - 2))
        fn = lambda x: _solve_coefficients(x, A_inv)
        for j in range(ndim - 2, -1, -1):
            ax = int(j >= axis)
            fn = vmap(fn, ax, ax)
        data = fn(data)
    return data


def _spline_basis(t: Array) -> Array:
    at = jnp.abs(t)
    fn1 = lambda t: (2 - t) ** 3
    fn2 = lambda t: 4 - 6 * t**2 + 3 * t**3
    return jnp.where(
        at >= 1, jnp.where(at <= 2, fn1(at), 0), jnp.where(at <= 1, fn2(at), 0)
    )


def _spline_value(coefficients: Array, coordinate: Array, indexes: Array) -> Array:
    coefficient = jnp.squeeze(
        lax.dynamic_slice(coefficients, indexes, [1] * coefficients.ndim)
    )
    fn = vmap(lambda x, i: _spline_basis(x - i + 1), (0, 0))
    return coefficient * fn(coordinate, indexes).prod()


def _spline_point(coefficients: Array, coordinate: Array) -> Array:
    index_fn = lambda x: (jnp.arange(0, 4) + jnp.floor(x)).astype(int)
    index_vals = vmap(index_fn)(coordinate)
    indexes = jnp.array(jnp.meshgrid(*index_vals, indexing="ij"))
    fn = lambda index: _spline_value(coefficients, coordinate, index)
    return vmap(fn)(indexes.reshape(coefficients.ndim, -1).T).sum()


def _cubic_spline(input: Array, coordinates: Array) -> Array:
    coefficients = _spline_coefficients(input)
    points = coordinates.reshape(input.ndim, -1).T
    fn = lambda coord: _spline_point(coefficients, coord)
    return vmap(fn)(points).reshape(coordinates.shape[1:])


@functools.partial(api.jit, static_argnums=(2, 3, 4))
def _map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str,
    cval: ArrayLike,
) -> Array:
    input_arr = jnp.asarray(input)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]
    cval = jnp.asarray(cval, input_arr.dtype)

    if len(coordinates) != input_arr.ndim:
        raise ValueError(
            "coordinates must be a sequence of length input.ndim, but {} != {}".format(
                len(coordinates), input_arr.ndim
            )
        )

    index_fixer = _INDEX_FIXERS.get(mode)
    if index_fixer is None:
        raise NotImplementedError(
            "map_coordinates does not yet support mode {}. "
            "Currently supported modes are {}.".format(mode, set(_INDEX_FIXERS))
        )

    if mode == "constant":
        is_valid = lambda index, size: (0 <= index) & (index < size)
    else:
        is_valid = lambda index, size: True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    elif order == 3:
        interp_fun = _cubic_indices_and_weights
    else:
        raise NotImplementedError("map_coordinates currently requires order<=1 or 3")

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = util.unzip3(items)
        if order == 3:
            if mode == "reflect":
                raise NotImplementedError(
                    "Cubic interpolation with mode='reflect' is not implemented."
                )
            interpolated = _cubic_spline(input_arr, jnp.array(indices))
            if all(valid is True for valid in validities):
                outputs.append(interpolated)
            else:
                all_valid = functools.reduce(operator.and_, validities)
                outputs.append(jnp.where(all_valid, interpolated, cval))
        else:
            if all(valid is True for valid in validities):
                contribution = input_arr[indices]
            else:
                all_valid = functools.reduce(operator.and_, validities)
                contribution = jnp.where(all_valid, input_arr[indices], cval)
            outputs.append(_nonempty_prod(weights) * contribution)
    result = _nonempty_sum(outputs)
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str = "constant",
    cval: ArrayLike = 0.0,
) -> Array:
    """Map coordinates using cubic spline interpolation.

    Args:
        input:
            The input array.
        coordinates:
            Sequence of coordinate arrays for each dimension.
        order:
            Interpolation order (0=nearest, 1=linear, 3=cubic).
        mode:
            How to handle out-of-bounds coordinates.
        cval:
            Value for out-of-bounds when mode='constant'.

    Returns:
        Interpolated values at the given coordinates.
    """
    return _map_coordinates(input, coordinates, order, mode, cval)


# =============================================================================
# Image Resampling
# =============================================================================


def ccw_rotation_matrix(rotation_deg: float) -> Array:
    """Return the counter-clockwise rotation matrix for a given angle.

    Args:
        rotation_deg:
            Rotation angle in degrees.

    Returns:
        2x2 rotation matrix.
    """
    theta = jnp.deg2rad(rotation_deg)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return jnp.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ]
    )


@functools.partial(api.jit, static_argnames=["shape_tgt"])
def resample_flux(
    f_src: Array,
    pixscale_src: float,
    pixscale_tgt: float,
    shape_tgt: tuple[int, int],
    rotation_deg: float = 0.0,
) -> Array:
    """Resample an image onto a new grid while conserving total flux.

    This function performs an affine transformation (rotation and scaling) to map
    the source image onto a target grid. It converts the source image to surface
    brightness (flux density), interpolates it at the target pixel coordinates,
    and then converts back to integrated flux per pixel.

    Args:
        f_src:
            The source image (2D array) containing integrated flux values per
            pixel. Shape: (ny_src, nx_src).
        pixscale_src:
            The pixel scale of the source image (e.g., arcsec/pixel or
            lambda/D). Must be in the same units as pixscale_tgt.
        pixscale_tgt:
            The pixel scale of the target image. Must be in the same units as
            pixscale_src.
        shape_tgt:
            The shape of the target image (ny_tgt, nx_tgt).
        rotation_deg:
            The angle to rotate the source image counter-clockwise in degrees.
            Default is 0.0.

    Returns:
        The resampled image on the target grid with total flux conserved.
        Shape: (ny_tgt, nx_tgt).
    """
    ny_src, nx_src = f_src.shape
    ny_tgt, nx_tgt = shape_tgt

    # Surface brightness (flux per unit area)
    s_src = f_src / (pixscale_src**2)

    # Affine matrix (TARGET pixel centres -> SOURCE coordinates)
    scale = pixscale_tgt / pixscale_src
    a_mat = ccw_rotation_matrix(rotation_deg) * scale

    c_src = jnp.array([(ny_src - 1) / 2.0, (nx_src - 1) / 2.0])
    c_tgt = jnp.array([(ny_tgt - 1) / 2.0, (nx_tgt - 1) / 2.0])
    offset = c_src - a_mat @ c_tgt

    # Grid of TARGET pixel centres
    y_coords = jnp.arange(ny_tgt)
    x_coords = jnp.arange(nx_tgt)
    y_tgt, x_tgt = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    # (2, ny_tgt, nx_tgt)
    coords = jnp.stack([y_tgt, x_tgt], axis=0)
    coords_src = (a_mat @ coords.reshape(2, -1) + offset[:, None]).reshape(coords.shape)

    # Interpolate surface brightness
    s_tgt = map_coordinates(
        s_src, [coords_src[0], coords_src[1]], order=3, mode="constant", cval=0.0
    )

    # Back to integrated flux per target pixel
    return s_tgt * (pixscale_tgt**2)


def downsample_psf(
    psf: Array,
    src_pixscale: float,
    target_shape: tuple[int, int],
) -> tuple[Array, float]:
    """Downsample a PSF to target shape while conserving total flux.

    Args:
        psf:
            The source PSF image (2D array).
        src_pixscale:
            The pixel scale of the source PSF (in lambda/D or other units).
        target_shape:
            The target shape (ny_tgt, nx_tgt).

    Returns:
        Tuple of (resampled_psf, new_pixscale).
    """
    ny_src, nx_src = psf.shape
    ny_tgt, nx_tgt = target_shape

    # Calculate new pixel scale (same field of view, fewer pixels)
    # Assumes square pixels and same scale in both dimensions
    scale_factor = ny_src / ny_tgt
    tgt_pixscale = src_pixscale * scale_factor

    # Resample using flux-conserving interpolation
    resampled = resample_flux(
        jnp.asarray(psf),
        src_pixscale,
        tgt_pixscale,
        target_shape,
        rotation_deg=0.0,
    )

    return resampled, tgt_pixscale


def downsample_psfs(
    psfs: np.ndarray,
    src_pixscale: float,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Downsample a stack of PSFs to target shape while conserving total flux.

    Args:
        psfs:
            Stack of PSF images with shape (N, H, W).
        src_pixscale:
            The pixel scale of the source PSFs (in lambda/D or other units).
        target_shape:
            The target shape (ny_tgt, nx_tgt) for each PSF.

    Returns:
        Tuple of (resampled_psfs as numpy array, new_pixscale).
    """
    ny_tgt, _ = target_shape

    # Calculate new pixel scale
    scale_factor = psfs.shape[1] / ny_tgt
    tgt_pixscale = src_pixscale * scale_factor

    # Ensure native byte order for JAX compatibility
    # FITS files often use big-endian (>f8) which JAX doesn't support
    if not psfs.dtype.isnative:
        psfs = psfs.astype(psfs.dtype.newbyteorder("="))

    # Vectorize over the PSF stack using vmap
    resample_single = lambda psf: resample_flux(
        psf,
        src_pixscale,
        tgt_pixscale,
        target_shape,
        rotation_deg=0.0,
    )

    # Use vmap for efficient batch processing
    resample_batch = vmap(resample_single)
    resampled_jax = resample_batch(jnp.asarray(psfs))

    # Convert back to numpy
    resampled_np = np.asarray(resampled_jax)

    return resampled_np, tgt_pixscale
