from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
import xarray as xr
from lod_unit import lod
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import rotate, zoom
from tqdm import tqdm

from yippy.logger import setup_logger


class Coronagraph:
    def __init__(self, yip_path, logging_level="INFO"):
        """
        Args:
            yip_path (str):
                Yield input package directory. Must have fits files
                    stellar_intens.fits - Stellar intensity map
                        Unitless 3d array of the stellar intensity function I,
                        as a function of (x, y) pixel coordinates and the
                        stellar angular diameter theta_star. Values in the map
                        are equal to the stellar count rate in a given pixel
                        divided by the total stellar count rate entering the
                        coronagraph. Does not include reductions such as QE, as
                        in without a coronagraph the total of I is unity.
                    stellar_intens_diam_list.fits - Stellar diameter list
                        A vector of stellar diameter values (lam/D) corresponding
                        to the theta_star values in stellar_intens.
                    offax_psf_offset_list - The off-axis PSF list
                    offax_psf - PSF of off-axis sources
                    sky_trans - Sky transmission data
            logging_level (str):
                Logging level for the logger (e.g. INFO, DEBUG, WARNING, ERROR,
                CRITICAL), use to suppress logging if used as part of a larger
                workflow. Default is INFO.
        """

        self.logger = setup_logger(logging_level)
        ###################
        # Read input data #
        ###################
        self.logger.info("Creating coronagraph")

        yip_path = Path(yip_path)
        self.name = yip_path.stem
        # Get header and calculate the lambda/D value
        stellar_intens_header = pyfits.getheader(
            Path(yip_path, "stellar_intens.fits"), 0
        )

        # Stellar intensity of the star being observed as function of stellar
        # angular diameter (unitless)
        self.stellar_intens = pyfits.getdata(Path(yip_path, "stellar_intens.fits"), 0)
        # the stellar angular diameters in stellar_intens_1 in units of lambda/D
        self.stellar_intens_diam_list = (
            pyfits.getdata(Path(yip_path, "stellar_intens_diam_list.fits"), 0) * lod
        )

        # Get pixel scale with units
        self.pixel_scale = stellar_intens_header["PIXSCALE"] * lod / u.pixel

        # Load off-axis data (e.g. the planet) (unitless intensity maps)
        self.offax_psf = pyfits.getdata(Path(yip_path, "offax_psf.fits"), 0)

        # The offset list here is in units of lambda/D
        self.offax_psf_offset_list = (
            pyfits.getdata(Path(yip_path, "offax_psf_offset_list.fits"), 0) * lod
        )

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################
        if len(self.offax_psf_offset_list.shape) > 1:
            if (self.offax_psf_offset_list.shape[1] != 2) and (
                self.offax_psf_offset_list.shape[0] == 2
            ):
                # This condition occurs when the offax_psf_offset_list is transposed
                # from the expected format for radially symmetric coronagraphs
                self.offax_psf_offset_list = self.offax_psf_offset_list.T

        # Check that we have both x and y offset information (even if there
        # is only one axis with multiple values)
        if self.offax_psf_offset_list.shape[1] != 2:
            raise UserWarning("Array offax_psf_offset_list should have 2 columns")

        # Get the unique values of the offset list so that we can format the
        # data into
        self.offax_psf_offset_x = np.unique(self.offax_psf_offset_list[:, 0])
        self.offax_psf_offset_y = np.unique(self.offax_psf_offset_list[:, 1])

        if (len(self.offax_psf_offset_x) == 1) and (
            self.offax_psf_offset_x[0] == 0 * lod
        ):
            self.type = "1d"
            # Instead of handling angles for 1dy, swap the x and y
            self.offax_psf_offset_x, self.offax_psf_offset_y = (
                self.offax_psf_offset_y,
                self.offax_psf_offset_x,
            )

            # self.offax_psf_base_angle = 90.0 * u.deg
            self.logger.info("Coronagraph is radially symmetric")
        elif (len(self.offax_psf_offset_y) == 1) and (
            self.offax_psf_offset_y[0] == 0 * lod
        ):
            self.type = "1d"
            # self.offax_psf_base_angle = 0.0 * u.deg
            self.logger.info("Coronagraph is radially symmetric")
        elif len(self.offax_psf_offset_x) == 1:
            # 1 dimensional with offset (e.g. no offset=0)
            self.type = "1dno0"
            self.offax_psf_offset_x, self.offax_psf_offset_y = (
                self.offax_psf_offset_y,
                self.offax_psf_offset_x,
            )
            # self.offax_psf_base_angle = 90.0 * u.deg
            self.logger.info("Coronagraph is radially symmetric")
        elif len(self.offax_psf_offset_y) == 1:
            self.type = "1dno0"
            # self.offax_psf_base_angle = 0.0 * u.deg
            self.logger.info("Coronagraph is radially symmetric")
        elif np.min(self.offax_psf_offset_list) >= 0 * lod:
            self.type = "2dq"
            # self.offax_psf_base_angle = 0.0 * u.deg
            # self.logger.info(
            #     f"Quarterly symmetric response --> reflecting PSFs ({self.type})"
            # )
            self.logger.info("Coronagraph is quarterly symmetric")
        else:
            self.type = "2df"
            # self.offax_psf_base_angle = 0.0 * u.deg
            self.logger.info("Coronagraph response is full 2D")

        ############
        # Clean up #
        ############
        # Center coronagraph model so that image size is odd and central pixel is center
        # TODO: Automate this process
        verified_coronagraph_models = [
            "LUVOIR-A_APLC_10bw_smallFPM_2021-05-05_Dyn10pm-nostaticabb",
            "LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb",
            "LUVOIR-B-VC6_timeseries",
            "LUVOIR-B_VC6_timeseries",
        ]
        if yip_path.parts[-1] in verified_coronagraph_models:
            self.stellar_intens = self.stellar_intens[:, 1:, 1:]
            self.offax_psf = self.offax_psf[:, :-1, 1:]
        else:
            raise UserWarning(
                "Please validate centering for this unknown coronagraph model"
            )

        # Simulation parameters
        self.yip_path = yip_path

        #########################################################################
        # Interpolate coronagraph model (in log space to avoid negative values) #
        #########################################################################
        # Fill value for interpolation
        fill = np.log(1e-100)

        # interpolate stellar data
        self.ln_stellar_intens_interp = interp1d(
            self.stellar_intens_diam_list,
            np.log(self.stellar_intens),
            kind="cubic",
            axis=0,
            bounds_error=False,
            fill_value=fill,
        )
        self.stellar_intens_interp = lambda stellar_diam: np.exp(
            self.ln_stellar_intens_interp(stellar_diam)
        )

        # interpolate planet data depending on type
        if "1" in self.type:
            # Always set up to interpolate along the x axis
            self.ln_offax_psf_interp = interp1d(
                self.offax_psf_offset_list[:, 0],
                np.log(self.offax_psf),
                kind="cubic",
                axis=0,
                bounds_error=False,
                fill_value=fill,
            )
        else:
            zz_temp = self.offax_psf.reshape(
                self.offax_psf_offset_x.shape[0],
                self.offax_psf_offset_y.shape[0],
                self.offax_psf.shape[1],
                self.offax_psf.shape[2],
            )
            if self.type == "2dq":
                # Reflect PSFs to cover the x = 0 and y = 0 axes.
                offax_psf_offset_x = np.append(
                    -self.offax_psf_offset_x[0], self.offax_psf_offset_x
                )
                offax_psf_offset_y = np.append(
                    -self.offax_psf_offset_y[0], self.offax_psf_offset_y
                )
                zz = np.pad(zz_temp, ((1, 0), (1, 0), (0, 0), (0, 0)))
                zz[0, 1:] = zz_temp[0, :, ::-1, :]
                zz[1:, 0] = zz_temp[:, 0, :, ::-1]
                zz[0, 0] = zz_temp[0, 0, ::-1, ::-1]

                self.ln_offax_psf_interp = RegularGridInterpolator(
                    (offax_psf_offset_x, offax_psf_offset_y),
                    np.log(zz),
                    method="linear",
                    bounds_error=False,
                    fill_value=fill,
                )
            else:
                # This section included references to non-class attributes for
                # offax_psf_offset_x and offax_psf_offset_y. I think it meant
                # to be the class attributes
                self.ln_offax_psf_interp = RegularGridInterpolator(
                    (self.offax_psf_offset_x, self.offax_psf_offset_y),
                    np.log(zz_temp),
                    method="linear",
                    bounds_error=False,
                    fill_value=fill,
                )
        self.offax_psf_interp = lambda coordinate: np.exp(
            self.ln_offax_psf_interp(coordinate)
        )

        ##################################################
        # Get remaining parameters and throughput values #
        ##################################################

        # Gets the number of pixels in the image
        self.img_pixels = self.stellar_intens.shape[1] * u.pixel
        self.npixels = self.img_pixels.value.astype(int)

        # Photometric parameters.
        head = pyfits.getheader(Path(yip_path, "stellar_intens.fits"), 0)

        # fractional obscuration
        self.frac_obscured = head["OBSCURED"]

        # fractional bandpass
        self.frac_bandwidth = (head["MAXLAM"] - head["MINLAM"]) / head["LAMBDA"]

        # PSF datacube info
        self.has_psf_datacube = False

    def get_coro_thruput(self, aperture_radius_lod=0.8, oversample=100, plot=True):
        """
        Get coronagraph throughput
        Args:
            aperture_radius (float):
                Circular aperture radius, in lambda/D (I think)
            oversample (int):
                Oversampling factor for interpolation
            plot (Boolean):
                Whether to plot the coronagraph throughput
        Returns:
            coro_thruput (float):
                Coronagraph throughput
        """
        # Add units
        aperture_radius = aperture_radius_lod * lod

        # Compute off-axis PSF at the median separation value
        # Previously was labeled half max, but there is no guarantee the
        # separations are equally spaced
        if len(self.offax_psf_offset_x) != 1:
            med_offset = self.offax_psf_offset_x[self.offax_psf_offset_x.shape[0] // 2]
        elif len(self.offax_psf_offset_y) != 1:
            med_offset = self.offax_psf_offset_y[self.offax_psf_offset_y.shape[0] // 2]
        else:
            raise UserWarning(
                (
                    "Array offax_psf_offset_list should have more than 1"
                    " unique element for at least one axis"
                )
            )
        # Create (x, y) coordiantes of the aperture in lam/D
        # if self.type in ["1dx", "1dxo"]:
        aperture_pos = u.Quantity([med_offset, self.offax_psf_offset_y[0]])
        # elif self.type in ["1dy", "1dyo"]:
        #     aperture_pos = u.Quantity([self.offax_psf_offset_x[0], med_offset])

        # Create image
        imgs = self.offax_psf_interp(med_offset)

        # Compute aperture position and radius on subarray in pixels.
        # This was 3 times the aperture radius in pixels, I don't know why 3
        # Npix = int(np.ceil(3 * aperture_radius / self.pixel_scale))
        aperture_radius_pix = np.ceil(
            3 * aperture_radius / self.pixel_scale
        ).value.astype(int)

        aperture_pos_pix = (
            (aperture_pos / self.pixel_scale).value + (imgs.shape[0] - 1) / 2
        ).astype(int)
        subarr = imgs[
            aperture_pos_pix[1] - aperture_radius_pix : aperture_pos_pix[1]
            + aperture_radius_pix
            + 1,
            aperture_pos_pix[0] - aperture_radius_pix : aperture_pos_pix[0]
            + aperture_radius_pix
            + 1,
        ]
        # (aperture_pos / self.pixel_scale + (imgs.shape[0] - 1) / 2.0)
        # This doesn't make sense to me
        pos_subarr = [0, 0] + aperture_radius_pix
        rad_subarr = aperture_radius / self.pixel_scale

        # Compute aperture position and radius on oversampled subarray in pixels.
        norm = np.sum(subarr)
        subarr_zoom = zoom(subarr, oversample, mode="nearest", order=5)
        subarr_zoom *= norm / np.sum(subarr_zoom)
        pos_subarr_zoom = pos_subarr * oversample + (oversample - 1.0) / 2.0
        rad_subarr_zoom = rad_subarr * oversample

        # Compute aperture on oversampled subarray in pixels.
        ramp = np.arange(subarr_zoom.shape[0])
        offax_psf_offset_x, yy = np.meshgrid(ramp, ramp)
        aptr = (
            np.sqrt(
                (offax_psf_offset_x - pos_subarr_zoom[0]) ** 2
                + (yy - pos_subarr_zoom[1]) ** 2
            )
            <= rad_subarr_zoom.value
        )

        # Compute coronagraph throughput
        coro_thruput = np.sum(subarr_zoom[aptr])

        return coro_thruput

    def get_disk_psfs(self):
        """
        Load the disk image from a file or generate it if it doesn't exist
        """
        # Load data cube of spatially dependent PSFs.
        disk_dir = Path(".cache/disks/")
        if not disk_dir.exists():
            disk_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            disk_dir,
            self.yip_path.name + ".nc",
        )

        coords = {
            "x psf offset (pix)": np.arange(self.npixels),
            "y psf offset (pix)": np.arange(self.npixels),
            "x (pix)": np.arange(self.npixels),
            "y (pix)": np.arange(self.npixels),
        }
        dims = ["x psf offset (pix)", "y psf offset (pix)", "x (pix)", "y (pix)"]
        if path.exists():
            self.logger.info(
                "Loading data cube of spatially dependent PSFs, please hold..."
            )
            psfs_xr = xr.open_dataarray(path)
        else:
            self.logger.info(
                "Calculating data cube of spatially dependent PSFs, please hold..."
            )
            # Compute pixel grid.
            # lambda/D
            pixel_lod = (
                (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                * u.pixel
                * self.pixel_scale
            )

            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")

            # lambda/D
            pixel_dist_lod = np.sqrt(x_lod**2 + y_lod**2)

            # deg
            pixel_angle = np.arctan2(y_lod, x_lod)

            # Compute pixel grid contrast.
            psfs_shape = (
                pixel_dist_lod.shape[0],
                pixel_dist_lod.shape[1],
                self.npixels,
                self.npixels,
            )
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            npsfs = np.prod(pixel_dist_lod.shape)

            pbar = tqdm(
                total=npsfs, desc="Computing datacube of PSFs at every pixel", delay=0.5
            )

            radially_symmetric_psf = "1d" in self.type
            # Get the PSF (npixel, npixel) of a source at every pixel

            # Note: intention is that i value maps to x offset and j value maps
            # to y offset
            for i in range(pixel_dist_lod.shape[0]):
                for j in range(pixel_dist_lod.shape[1]):
                    # Basic structure here is to get the distance in lambda/D,
                    # determine whether the psf has to be rotated (if the
                    # coronagraph is defined in 1 dimension), evaluate
                    # the offaxis psf at the distance, then rotate the
                    # image
                    if self.type == "1d":
                        psf_eval_dists = pixel_dist_lod[i, j]
                        rotate_angle = pixel_angle[i, j]
                    elif self.type == "1dno0":
                        psf_eval_dists = np.sqrt(
                            pixel_dist_lod[i, j] ** 2 - self.offax_psf_offset_x[0] ** 2
                        )
                        rotate_angle = pixel_angle[i, j] + np.arcsin(
                            self.offax_psf_offset_x[0] / pixel_dist_lod[i, j]
                        )
                    elif self.type == "2dq":
                        # lambda/D
                        temp = np.array([y_lod[i, j], x_lod[i, j]])
                        psf = self.offax_psf_interp(np.abs(temp))[0]
                        if y_lod[i, j] < 0.0:
                            # lambda/D
                            psf = psf[::-1, :]
                        if x_lod[i, j] < 0.0:
                            # lambda/D
                            psf = psf[:, ::-1]
                    else:
                        # lambda/D
                        temp = np.array([y_lod[i, j], x_lod[i, j]])
                        psf = self.offax_psf_interp(temp)[0]

                    if radially_symmetric_psf:
                        psf = self.ln_offax_psf_interp(psf_eval_dists)
                        temp = np.exp(
                            rotate(
                                psf,
                                -rotate_angle.to(u.deg).value,
                                reshape=False,
                                mode="nearest",
                                order=5,
                            )
                        )
                    psfs[i, j] = temp
                    pbar.update(1)

            # Save data cube of spatially dependent PSFs.
            psfs_xr = xr.DataArray(
                psfs,
                coords=coords,
                dims=dims,
            )
            psfs_xr.to_netcdf(path)
        self.has_psf_datacube = True
        self.psf_datacube = np.ascontiguousarray(psfs_xr)
