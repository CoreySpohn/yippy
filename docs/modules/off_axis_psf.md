# Off-Axis PSF Files Documentation
## Introduction

This documentation provides detailed information about the off-axis Point
Spread Function (PSF) files used in the `yippy` package, designed for working
with coronagraph simulations. The off-axis PSF maps are crucial for
understanding the optical system's response to light sources that are not
aligned with the optical axis, typically representing exoplanets or other
celestial bodies in astrophysical observations.

## File Description

The `offax_psf.fits` file contains a 3D array of off-axis PSF maps. These maps
are indexed by two spatial coordinates (x, y) and a third index representing
different astrophysical offsets (from the `offax_psf_offset_list.fits` file).
These offsets correspond to the position of a point source, measured in units
of λ/D, where λ is the wavelength of observation and D is the diameter of the
telescope's primary mirror.

### File Components
- Offset Dimension: Each slice along this dimension corresponds to a PSF shifted by a specific astrophysical offset, facilitating the simulation of various positions of celestial bodies.
- Spatial Dimensions (x, y): Represents the 2D PSF as pixels.

The `offax_psf_offset_list.fits` file accompanies the PSF maps, providing a 2xN_offsets array that lists the (x, y) values for each offset in the same units.
## Usage Tutorial
### With `yippy`
```python
from yippy import Coronagraph

coro = Coronagraph("path/to/yield/input/package")
psf = coro.offax(x, y)
```
