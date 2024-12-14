<p align="center">
  <img width = 250 src="https://raw.githubusercontent.com/coreyspohn/yippy/main/docs/_static/logo.png" alt="lod_unit logo" />
  <br><br>
</p>

<p align="center">
  <a href="https://codecov.io/gh/CoreySpohn/lod_unit"><img src="https://img.shields.io/codecov/c/github/coreyspohn/lod_unit?token=UCUVYCRWVG&style=flat-square&logo=codecov" alt="Codecov"/></a>
  <a href="https://pypi.org/project/lod_unit/"><img src="https://img.shields.io/pypi/v/lod_unit.svg?style=flat-square" alt="PyPI"/></a>
  <a href="https://lod-unit.readthedocs.io"><img src="https://readthedocs.org/projects/lod_unit/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/coreyspohn/lod_unit/actions/workflows/ci.yml/"><img src="https://img.shields.io/github/actions/workflow/status/coreyspohn/lod_unit/ci.yml?branch=main&logo=github&style=flat-square" alt="CI"/></a>
</p>

---

# yippy

A minimal wrapper to create a coronagraph object from a yield input package.
A core feature is its ability to use Fourier interpolation to generate off axis
PSFs at arbitrary locations in the (x,y) plane efficiently.
Uses [JAX](https://jax.readthedocs.io/en/latest/) to speed up computation by default,
with an optional Python backend.

## Installation

```bash
pip install yippy
```

## Use

Typical use will look like

```python
import astropy.units as u
from lod_unit import lod
from yippy import Coronagraph

# Create a coronagraph object by specifying the path to the yield input package
aplc = Coronagraph(Path("input/ApodSol_APLC"))

# Offaxis PSF at a given point source position in the (x,y) plane
x_pos = 2 * lod # 2 lambda/D
y_pos = 5 * lod # 5 lambda/D
offaxis_psf = aplc.offax(x_pos, y_pos)

# On-axis intensity map with a stellar diameter
stellar_diameter = 1*lod
stellar_intensity = aplc.stellar_intensity(stellar_diameter)

# Sky transmission map for extended sources
sky_trans = aplc.sky_trans()
```

### Units

Yield input packages are given in $`\lambda / D`$ units so `yippy` treats them
as the default and uses the `lod_unit` package to define the `lod` unit. However,
it can use pixels (_coronagraph pixels_), angular separation (angle units), or
apparent separation (length units).

```python
# pixels
x_pos = 2 * u.pix
y_pos = 5 * u.pix
offaxis_psf = aplc.offax(x_pos, y_pos)

# angular separation
telescope_diameter = 10 * u.m
wavelength = 500 * u.nm
offaxis_psf = aplc.offax(x_pos, y_pos, lam=wavelength, D=telescope_diameter)

# apparent separation
star_dist = 10 * u.pc
offaxis_psf = aplc.offax(x_pos, y_pos, lam=wavelength, D=telescope_diameter, dist=star_dist)
```
