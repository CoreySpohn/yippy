# Changelog

## [1.5.0](https://github.com/CoreySpohn/yippy/compare/v1.4.0...v1.5.0) (2024-12-02)


### Features

* Add OffJAx class ([46c82f9](https://github.com/CoreySpohn/yippy/commit/46c82f92e0957924dc3e8be70dba199d3917b8ea))
* Make x and y symmetry optional, remove rotational symmetry ([fe1cb33](https://github.com/CoreySpohn/yippy/commit/fe1cb3346feff5aedd43a649bfde273672b150c7))


### Bug Fixes

* Added x/y symmetry options to the JAX implementation ([08d08e6](https://github.com/CoreySpohn/yippy/commit/08d08e62e70d1533c3b0644560a5ae7569cd570f))

## [1.4.0](https://github.com/CoreySpohn/yippy/compare/v1.3.0...v1.4.0) (2024-08-30)


### Features

* Add expressive logger ([4eec73c](https://github.com/CoreySpohn/yippy/commit/4eec73c74168b1afd8246919ca05d43cf9e6bb7f))
* Add Fourier interpolation utility functions ([d023e1c](https://github.com/CoreySpohn/yippy/commit/d023e1c650d674829c4117738b05f6816ff2762f))
* Implement FFT based interpolation and rotation ([8ecac66](https://github.com/CoreySpohn/yippy/commit/8ecac660316e9f003f27d997874fc0ebcd5202e9))
* Implemented fft interpolation in the One-D case ([5309cb6](https://github.com/CoreySpohn/yippy/commit/5309cb64ef29819831aad53723b344a3132c3ebc))


### Bug Fixes

* Fix the import of the logger ([4021431](https://github.com/CoreySpohn/yippy/commit/4021431b9a131fd703b3fc154c40f394c145dbf7))
* **main:** Improve the one D PSF to only take the log if necessary ([845fdf5](https://github.com/CoreySpohn/yippy/commit/845fdf53d759894ae17b40e1fff1b689acbb49a8))

## [1.3.0](https://github.com/CoreySpohn/yippy/compare/v1.2.0...v1.3.0) (2024-04-23)


### Features

* **main:** Add a temporary sky_trans file ([ad89135](https://github.com/CoreySpohn/yippy/commit/ad89135fc2687b60af018e7a9fde503513ee1854))
* **main:** Added dataclass that handles the header ([3120eda](https://github.com/CoreySpohn/yippy/commit/3120eda53bb75dc96ead74ae3e37c5cd206785ac))

## [1.2.0](https://github.com/CoreySpohn/yippy/compare/v1.1.1...v1.2.0) (2024-04-17)


### Features

* **main:** Added stellar intensity map ([481d333](https://github.com/CoreySpohn/yippy/commit/481d333b89280a906bf8be3642f0eb7bf1fa946e))
* **main:** Adding more support for 2d and quarter symmetric coronagraphs ([3e98780](https://github.com/CoreySpohn/yippy/commit/3e9878034b37535780ee0004f69ad4409b961445))


### Bug Fixes

* **main:** Fixed error in how the quarter symmetric PSFs handled 0*lam/D values ([3e6943f](https://github.com/CoreySpohn/yippy/commit/3e6943f6bfaf89c8b8ba353921bc5a245696e194))

## [1.1.1](https://github.com/CoreySpohn/yippy/compare/v1.1.0...v1.1.1) (2024-04-05)


### Bug Fixes

* **main:** Fixed handling when given single dimensional offax_psf_offsets_list without a second column of zeros ([86f0cc7](https://github.com/CoreySpohn/yippy/commit/86f0cc795d6471b8abaddc3e80278d97aaf93706))

## [1.1.0](https://github.com/CoreySpohn/yippy/compare/v1.0.0...v1.1.0) (2024-04-05)


### Features

* **main:** Add off-axis psfs with automatic unit conversion ([6f5b815](https://github.com/CoreySpohn/yippy/commit/6f5b815093e6fe7898cd625451ad31ab1acee221))

## 1.0.0 (2024-03-22)


### Features

* Automatic versioning and changelog ([ef1acc1](https://github.com/CoreySpohn/yippy/commit/ef1acc1381058fdb32f6b32bb3d695a2035ad048))


### Bug Fixes

* Adding pre-commit hook for conventional commit formatting ([3b52ed6](https://github.com/CoreySpohn/yippy/commit/3b52ed6e3233b7acaa51f5ee8cd2a2b3f317912f))
* putting the workflows in the right folder ought to help ([ff1bf0a](https://github.com/CoreySpohn/yippy/commit/ff1bf0a12850691de801c9a3ba4202f3e8f4f7f1))
