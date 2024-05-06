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




- - -

# yippy

A minimal wrapper to create a coronagraph object from a yield input package.

## Installation
```bash
pip install yippy
```
## Use
Typical use will look like
```python
from lod_unit import lod
from yippy import Coronagraph

aplc = Coronagraph(Path("input/ApodSol_APLC"))

# Offaxis PSF at a given point source position
point_source_position = [2, 5]*lod
offaxis_psf = aplc.offax(**point_source_position)

# On-axis intensity map with a stellar diameter
stellar_diameter = 1*lod
stellar_intensity = aplc.stellar_intensity(stellar_diameter)

# Sky transmission map for extended sources
sky_trans = aplc.sky_trans()
```
