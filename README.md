<p align="center">
  <img width = 250 src="https://raw.githubusercontent.com/coreyspohn/yippy/main/docs/_static/logo.png" alt="yippy logo" />
  <br><br>
</p>

<p align="center">
  <a href="https://pypi.org/project/yippy/"><img src="https://img.shields.io/pypi/v/yippy.svg?style=flat-square" alt="PyPI"/></a>
  <a href="https://yippy.readthedocs.io"><img src="https://readthedocs.org/projects/yippy/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <!-- <a href="https://github.com/coreyspohn/yippy/actions/workflows/ci.yml/"><img src="https://img.shields.io/github/actions/workflow/status/coreyspohn/yippy/ci.yml?branch=main&logo=github&style=flat-square" alt="CI"/></a> -->
</p>

---

# yippy

A wrapper to create a coronagraph object from a yield input package (a "YIP").
A core feature is its ability to use Fourier interpolation to generate off axis
PSFs at arbitrary locations in the (x,y) plane efficiently.
`yippy` uses [JAX](https://jax.readthedocs.io/en/latest/) to speed up
computation by default, with an optional Python backend.

## Installation

```bash
pip install yippy
```

## Use

Typical use will look like

```python
from lod_unit import lod
from yippy import Coronagraph

# Create a coronagraph object by specifying the path to the yield input package
coro = Coronagraph("input/LUVOIR_VVC")

# Off-axis PSF at a given point source position in the (x,y) plane
x_pos = 2 * lod # 2 lambda/D
y_pos = 5 * lod # 5 lambda/D
offaxis_psf = coro.offax(x_pos, y_pos)

# On-axis intensity map with a stellar diameter
stellar_diameter = 1*lod
stellar_intensity = coro.stellar_intensity(stellar_diameter)

# Sky transmission map for extended sources
sky_trans = coro.sky_trans()
```

### Units

Yield input packages use $`\lambda / D`$ units so `yippy` treats them
as the default and uses the `lod_unit` package to define the `lod` unit. However,
it can use three different `astropy` units: pixels (as defined by the yield
input package), angular separation (angle units), or apparent separation
(length units). If no units are provided it assumes the input is in $`\lambda / D`$.

```python
import astropy.units as u
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

## JAX

The default backend is JAX, which is a high-performance numerical computing library
that we use for JIT compilation and GPU/TPU support. By default, JAX uses 32-bit
floating point precision, which leads to faster computation and lower memory overhead
but results in lower precision (~1e-6 precision). If you need precision at the
1e-16 level, set `use_x64=True`.

### Off-axis PSF options

- `use_jax`: Use JAX for computation. Default is `True`.
- `use_x64`: Use 64-bit floating point precision. Default is `False`.
- `x_symmetric`: Off-axis PSF is symmetric about the x-axis. Default is `True`.
- `y_symmetric`: Off-axis PSF is symmetric about the y-axis. Default is `False`.
- `cpu_cores`: Number of CPU cores to use. Default is `1`.
- `platform`: Computing platform to use for JAX computation. Options are `cpu`, `gpu`, `tpu`. Default is `cpu`.

### Parallel processing of off-axis PSFs

The base call of `coronagraph.offax(x,y)` is the most user-friendly, but is not
the most efficient. When generating many PSFs it is recommended to convert all
required (x,y) positions into arrays of floats (in $`\lambda / D`$) and use the
`coronagraph.offax.create_psfs_parallel(x_arr, y_arr)` function. This function
uses JAX's `shard_map` to distribute the computation across multiple devices or
CPU cores.
