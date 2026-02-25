<p align="center">
  <img width = 250 src="https://raw.githubusercontent.com/coreyspohn/yippy/main/docs/_static/logo.png" alt="yippy logo" />
  <br><br>
</p>

<p align="center">
  <a href="https://pypi.org/project/yippy/"><img src="https://img.shields.io/pypi/v/yippy.svg?style=flat-square" alt="PyPI"/></a>
  <a href="https://yippy.readthedocs.io"><img src="https://readthedocs.org/projects/yippy/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/coreyspohn/yippy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/coreyspohn/yippy?style=flat-square" alt="License"/></a>
  <a href="https://pypi.org/project/yippy/"><img src="https://img.shields.io/pypi/pyversions/yippy?style=flat-square" alt="Python"/></a>
  <!-- <a href="https://github.com/coreyspohn/yippy/actions/workflows/ci.yml/"><img src="https://img.shields.io/github/actions/workflow/status/coreyspohn/yippy/ci.yml?branch=main&logo=github&style=flat-square" alt="CI"/></a> -->
</p>

---

# yippy

A Python and JAX library for loading coronagraph yield input packages (YIPs)
and computing coronagraph performance metrics. yippy provides Fourier-based
off-axis PSF interpolation, throughput/contrast/core-area curves, 2D
performance maps, and export to EXOSIMS and AYO formats.

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

# 2D performance maps (pixel grids)
throughput_map = coro.throughput_map()
core_area_map  = coro.core_area_map()
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

### `Coronagraph` -- Analysis & Data Management

The primary class for loading YIPs, computing performance curves, and
exporting to external formats:

```python
from yippy import Coronagraph

coro = Coronagraph("path/to/yip")

# 1D performance curves (scalar or array separations in lam/D)
coro.throughput(5.0)
coro.raw_contrast(5.0)
coro.occulter_transmission(5.0)
coro.core_area(5.0)
coro.core_mean_intensity(5.0)

# Noise floors
coro.noise_floor_exosims(5.0)           # |raw_contrast| / ppf
coro.noise_floor_ayo(5.0, ppf=30.0)     # core_mean_intensity / ppf

# 2D maps (full pixel grids)
coro.separation_map()
coro.throughput_map()
coro.core_area_map()
coro.core_mean_intensity_map()
coro.noise_floor_ayo_map(ppf=30.0)

# Export
coro.to_exosims()
coro.dump_ayo_csv("output.csv")
```

### `EqxCoronagraph` -- JIT-Compatible Simulation

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
    compute_truncation_throughput_curve,     # PSF truncation-ratio aperture
    compute_truncation_core_area_curve,
)

# Compute individual curves
separations, throughputs = compute_throughput_curve(coro)
separations, contrasts  = compute_raw_contrast_curve(coro)
```

### PSF Truncation Ratio

When `psf_trunc_ratio` is set (e.g. `Coronagraph(path, psf_trunc_ratio=0.3)`),
throughput and core area are computed using an adaptive aperture that includes
all oversampled pixels exceeding `ratio * peak`. This matches AYO's
`photap_frac` / `omega_lod` calculation and is recommended for ETC integration.

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

yippy uses [JAX](https://jax.readthedocs.io) for JIT compilation and
GPU/TPU-accelerated PSF generation. JAX defaults to 32-bit precision;
if you need 64-bit precision, configure it
**before** importing yippy or any other JAX-based library:

```python
# At the very top of your script
from hwoutils import enable_x64, set_platform

enable_x64()           # switch to float64
set_platform("cpu")    # or "gpu", "gpu,cpu" for fallback

# Now it's safe to import yippy
from yippy import Coronagraph
```

Or via environment variables (safest):

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python my_script.py
```

See the [JAX Configuration Guide](https://github.com/CoreySpohn/hwoutils/blob/main/docs/jax_configuration.md)
in hwoutils for details and common gotchas.

### Constructor options

- `use_jax`: Use JAX for PSF computation. Default is `True`.
- `x_symmetric`: Off-axis PSFs are symmetric about the x-axis. Default is `True`.
- `y_symmetric`: Off-axis PSFs are symmetric about the y-axis. Default is `True`.
- `cpu_cores`: Number of CPU cores for parallel PSF generation via `shard_map`. Default is `4`.

### Parallel processing of off-axis PSFs

The base call of `coronagraph.offax(x,y)` is the most user-friendly, but is not
the most efficient. When generating many PSFs it is recommended to convert all
required (x,y) positions into arrays of floats (in $`\lambda / D`$) and use the
`coronagraph.offax.create_psfs_parallel(x_arr, y_arr)` function. This function
uses JAX's `shard_map` to distribute the computation across multiple devices or
CPU cores.
