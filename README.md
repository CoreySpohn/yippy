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

## Quick Start

```python
from yippy import Coronagraph
from yippy.datasets import fetch_coronagraph

# Download an example YIP (cached after first call)
yip_path = fetch_coronagraph()

# Create a coronagraph object
coro = Coronagraph(yip_path)

# Off-axis PSF at a given (x, y) position
from lod_unit import lod
offaxis_psf = coro.offax(2 * lod, 5 * lod)

# Performance metrics at any separation
throughput = coro.throughput(5.0)        # scalar or array
contrast   = coro.raw_contrast(5.0)
occ_trans  = coro.occulter_transmission(5.0)
```

## Two-Class Design

yippy provides two coronagraph classes for different use cases:

| | `Coronagraph` | `EqxCoronagraph` |
|---|---|---|
| **Purpose** | Full-featured analysis & export | JIT-compiled simulation |
| **Backend** | NumPy/SciPy + JAX | Pure JAX/Equinox |
| **JIT-compatible** | No | Yes (`eqx.filter_jit`) |
| **GPU/TPU support** | PSF generation only | Everything |
| **I/O & export** | EXOSIMS FITS, AYO CSV | None (simulation only) |
| **Performance curves** | Computed on init | Converted from `Coronagraph` |

### `Coronagraph` — Analysis & Data Management

The primary class for loading YIPs, computing performance curves, and
exporting to external formats:

```python
from yippy import Coronagraph

coro = Coronagraph("path/to/yip")

# Access pre-computed performance curves
coro.throughput(5.0)
coro.raw_contrast(5.0)
coro.noise_floor(5.0)
coro.occulter_transmission(5.0)
coro.core_area(5.0)
coro.core_mean_intensity(5.0)

# Export to EXOSIMS format
coro.to_exosims()

# Export to AYO CSV format
coro.dump_ayo_csv("output.csv")
```

### `EqxCoronagraph` — JIT-Compatible Simulation

A pure JAX/Equinox module for use inside `jax.jit`-compiled pipelines:

```python
from yippy import EqxCoronagraph
import equinox as eqx

# Create from a YIP path directly
coro = EqxCoronagraph("path/to/yip")

# All methods are JIT-traceable
@eqx.filter_jit
def simulate(coro, x, y):
    psf = coro.create_psf(x, y)
    stellar = coro.stellar_intens(0.01)
    throughput = coro.throughput(5.0)
    return psf, stellar, throughput
```

## Performance Metrics

Individual metric functions are available in `yippy.performance` for
standalone analysis:

```python
from yippy.performance import (
    compute_throughput_curve,
    compute_raw_contrast_curve,
    compute_core_area_curve,
    compute_occ_trans_curve,
    compute_core_mean_intensity_curve,
)

# Compute individual curves
separations, throughputs = compute_throughput_curve(coro)
separations, contrasts  = compute_raw_contrast_curve(coro)
```

These are the same functions used internally by `Coronagraph` during
initialization.

## Example Data

yippy ships with `pooch`-managed example data for testing and notebooks:

```python
from yippy.datasets import fetch_coronagraph

# Downloads and caches an example apodized vortex coronagraph
yip_path = fetch_coronagraph()  # "eac1_aavc_512"
```

## Units

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
offaxis_psf = coro.offax(x_pos, y_pos)

# angular separation
telescope_diameter = 10 * u.m
wavelength = 500 * u.nm
offaxis_psf = coro.offax(x_pos, y_pos, lam=wavelength, D=telescope_diameter)

# apparent separation
star_dist = 10 * u.pc
offaxis_psf = coro.offax(x_pos, y_pos, lam=wavelength, D=telescope_diameter, dist=star_dist)
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
