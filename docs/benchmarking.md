# Benchmarking: Yippy vs AYO

This page summarizes validation of yippy's coronagraph performance metrics
against AYO's IDL implementation, tested on two coronagraphs (AAVC and OVC)
with PSF truncation ratio 0.3.

## Key Results

| Metric | AAVC mean (max err) | OVC mean (max err) |
|---|---|---|
| Core throughput | 0.9991 (10.3%) | 1.0002 (9.0%) |
| Sky transmission | 0.9994 (0.9%) | 0.9995 (1.0%) |
| Raw contrast | 1.0000 (< 0.1%) | 1.0000 (< 0.1%) |
| Noise floor | 1.0000 (< 0.1%) | 1.0000 (< 0.1%) |

The core-throughput extremes lie within 3 $\lambda/D$ of the image edge
(30 to 32 $\lambda/D$), where the core is cut off by the image border and
yippy's curve falls up to 10% below AYO's. Out to 29 $\lambda/D$ the core
throughput agrees to 2.0% (AAVC) and 1.1% (OVC), with means of 1.0031 and
1.0036; the largest difference is near the inner working angle, where yippy
measures the core of the PSF synthesized at each separation and AYO
interpolates linearly between the tabulated offsets.

Raw contrast and noise floor match exactly because both AYO and yippy
clamp values at the contrast floor ($10^{-10}$), which dominates across
most of the science region. The figure below focuses on throughput and
sky transmission, the two metrics with meaningful variation.

![Yippy vs AYO validation](_static/fig_benchmarking_all_metrics.png)
