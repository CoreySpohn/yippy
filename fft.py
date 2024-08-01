"""Fourier transform related functions."""

from pathlib import Path

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

# import jax.numpy as np
import numpy as np
from jax.numpy.fft import fft, fftfreq, fftshift, ifft


def frame_center(array, verbose=False):
    """Return the coordinates y,x of the frame(s) center.

    If odd: dim/2-0.5
    If even: dim/2

    Parameters
    ----------
    array : 2d/3d/4d numpy ndarray
        Frame or cube.
    verbose : bool optional
        If True the center coordinates are printed out.

    Returns:
    -------
    cy, cx : int
        Coordinates of the center.

    """
    if array.ndim == 2:
        shape = array.shape
    elif array.ndim == 3:
        shape = array[0].shape
    elif array.ndim == 4:
        shape = array[0, 0].shape
    else:
        raise ValueError("`array` is not a 2d, 3d or 4d array")

    cy = shape[0] / 2
    cx = shape[1] / 2

    if shape[0] % 2:
        cy -= 0.5
    if shape[1] % 2:
        cx -= 0.5

    if verbose:
        print("Center px coordinates at x,y = ({}, {})".format(cx, cy))

    return int(cy), int(cx)


def rotate_fft(array, angle):
    """Rotate a frame or 2D array using Fourier transforms.

    Rotation is equivalent to 3 consecutive linear shears, or 3 consecutive 1D
    FFT phase shifts. See details in [LAR97]_.

    Parameters
    ----------
    array : numpy ndarray
        Input image, 2d array.
    angle : float
        Rotation angle.

    Returns:
    -------
    array_out : numpy ndarray
        Resulting frame.

    Note:
    ----
    This method is slower than interpolation methods (e.g. opencv/lanczos4 or
    ndimage), but preserves the flux better (by construction it preserves the
    total power). It is more prone to large-scale Gibbs artefacts, so make sure
    no sharp edge nor bad pixels are present in the image to be rotated.

    Note:
    ----
    Warning: if input frame has even dimensions, the center of rotation
    will NOT be between the 4 central pixels, instead it will be on the top
    right of those 4 pixels. Make sure your images are centered with
    respect to that pixel before rotation.

    """
    y_ori, x_ori = array.shape

    # Cut the angle to [0, 360)
    angle = angle % 360
    # while angle < 0:
    #     angle += 360
    # while angle > 360:
    #     angle -= 360

    # # first convert to odd size before multiple 90deg rotations
    # if not y_ori % 2 or not x_ori % 2:
    #     array_in = np.zeros([array.shape[0] + 1, array.shape[1] + 1])
    #     array_in[:-1, :-1] = array.copy()
    #     # array_in = array_in.at[:-1, :-1].set(array.copy())
    # else:
    #     array_in = array.copy()

    # Number of 90deg rotations needed
    nrot90 = np.rint(angle / 90)

    # Cut the FFT rotation angle to [0, 90)
    dangle = angle % 90

    # Cut the FFT rotation angle to (-45, 45]
    if dangle > 45:
        # Any angle over 45 deg is equivalent to rotating by 90 - angle in
        # the opposite direction and then add an extra 90 deg rotation
        dangle = -(90 - dangle)
        nrot90 += 1

    # remove last row and column to make it even size before FFT
    # array_in = array_in[:-1, :-1]

    # Ensure the array is even size
    if y_ori % 2 or x_ori % 2:
        array_in = array.copy()
    else:
        # Add a row and column to make it even size
        array_in = np.zeros([array.shape[0] + 1, array.shape[1] + 1])
        array_in[:-1, :-1] = array.copy()
    a = np.tan(np.deg2rad(dangle) / 2)
    b = -np.sin(np.deg2rad(dangle))
    # a = np.tan(np.deg2rad(angle) / 2)
    # b = -np.sin(np.deg2rad(angle))

    ori_y, ori_x = array_in.shape

    cy, cx = frame_center(array)
    arr_xy = np.mgrid[0:ori_y, 0:ori_x]
    arr_y = arr_xy[0] - cy
    arr_x = arr_xy[1] - cx

    # TODO: make FFT padding work for other option than '0'.
    s_x = _fft_shear(array_in, arr_x, a, ax=1)
    s_xy = _fft_shear(s_x, arr_y, b, ax=0)
    s_xyx = _fft_shear(s_xy, arr_x, a, ax=1)

    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(15, 15))
    axes[0].imshow(np.real(array_in), origin="lower")
    axes[1].imshow(np.real(s_x), origin="lower")
    axes[2].imshow(np.real(s_xy), origin="lower")
    axes[3].imshow(np.real(s_xyx), origin="lower")
    plt.show()
    breakpoint()

    if y_ori % 2 or x_ori % 2:
        # set it back to original dimensions
        array_out = np.zeros([s_xyx.shape[0] + 1, s_xyx.shape[1] + 1])
        array_out[:-1, :-1] = np.real(s_xyx)
    else:
        array_out = np.real(s_xyx)

    return array_out


def _fft_shear(arr, arr_ori, c, ax, pad=0, shift_ini=True):
    ax2 = 1 - ax % 2
    freqs = fftfreq(arr_ori.shape[ax2])
    sh_freqs = fftshift(freqs)
    arr_u = np.tile(sh_freqs, (arr_ori.shape[ax], 1))
    if ax == 1:
        arr_u = arr_u.T
    s_x = fftshift(arr)
    s_x = fft(s_x, axis=ax)
    s_x = fftshift(s_x)
    s_x = np.exp(-2j * np.pi * c * arr_u * arr_ori) * s_x
    s_x = fftshift(s_x)
    s_x = ifft(s_x, axis=ax)
    s_x = fftshift(s_x)

    return s_x


# https://vip.readthedocs.io/en/latest/_modules/vip_hci/preproc/derotation.html#rotate_fft
if __name__ == "__main__":
    yip_dir = Path("input/usort_offaxis_optimal_order_6")
    offax_offsets_file = "offax_psf_offset_list.fits"
    offax_data_file = "offax_psf.fits"
    offsets = np.array(pyfits.getdata(Path(yip_dir, offax_offsets_file), 0))
    psfs = np.array(pyfits.getdata(Path(yip_dir, offax_data_file), 0))

    psf = psfs[20]
    psf_rot = rotate_fft(psf, 10)
    # n_angles = 9
    # fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))
    n_angles = 25
    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(15, 15))
    angles = np.linspace(0, 360, n_angles)

    for ax, angle in zip(axes.flatten(), angles):
        rot_psf = rotate_fft(psf, angle)
        ax.imshow(rot_psf, origin="lower")
        ax.set_title(f"Angle: {angle}")
    # axes[0].imshow(psf, origin="lower")
    # axes[1].imshow(rot_psf, origin="lower")
    # axes[0].set_title("Original PSF")
    # axes[1].set_title("Rotated PSF")
    plt.show()
    breakpoint()
