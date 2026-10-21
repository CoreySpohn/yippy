"""Write reproducible aperture accuracy tables for the public AAVC and Airy PSFs.

Run from the repository root with the repository on PYTHONPATH::

    python tests/test_aperture_convergence.py results/aperture-convergence

The old circle column executes performance.py from commit 3fc8081 at 2x
oversampling. The analytic Airy reference is Born and Wolf, Principles of
Optics, 7th edition, section 8.5.2, as documented in the test helpers.
"""

import argparse
import csv
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.special import j1
from test_band_limited_apertures import airy_psf, encircled_energy, image_coro

from yippy import Coronagraph, fetch_yip
from yippy import performance as perf


def write_table(directory, name, header, rows):
    """Persist numeric evidence as a CSV with units in the column names."""
    with (directory / name).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def historical_performance(revision):
    """Load an immutable historical module without modifying the checkout."""
    module = types.ModuleType(f"yippy.performance_{revision}")
    module.__package__ = "yippy"
    sys.modules[module.__name__] = module
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/yippy/performance.py"], text=True
    )
    exec(compile(source, f"performance@{revision}", "exec"), module.__dict__)
    return module


def measure(psf, pixel_lod, factor, ratio, center):
    """Return circle flux/area and truncation flux/area at one resolution."""
    with jax.enable_x64():
        image = jnp.asarray(psf, dtype=jnp.float64)
        cf, ca, _ = perf.circular_aperture(
            image,
            0.7 / pixel_lod,
            pixel_lod,
            oversample=factor,
            half_width_pix=int(np.ceil(1.7 / pixel_lod)) + 1,
            nominal_center=center,
        )
        tf, ta = perf.truncation_core(
            image,
            ratio,
            pixel_lod,
            oversample=factor,
            half_width_pix=perf.core_window_half_width_pix(pixel_lod),
            nominal_center=center,
        )
        return [float(v) for v in (cf, ca, tf, ta)]


def main(directory):
    """Measure all requested gates, including every native shipped offset."""
    directory.mkdir(parents=True, exist_ok=True)
    half_max = brentq(lambda r: (2 * j1(np.pi * r) / (np.pi * r)) ** 2 - 0.5, 0.1, 1)
    airy_rows = []
    for pixel_lod in (0.25, 0.5):
        for shift in (0.0, 0.3):
            psf = airy_psf(pixel_lod, shift)
            coro = image_coro(psf, pixel_lod)
            for radius in (0.7, 1.0, 1.22):
                flux = perf.compute_throughput_curve(coro, radius)[1][0]
                airy_rows.append(
                    [
                        pixel_lod,
                        shift,
                        "circle_flux",
                        radius,
                        flux,
                        encircled_energy(radius),
                    ]
                )
            values = measure(psf, pixel_lod, 10, 0.5, (128, 128))
            for metric, measured, analytic in (
                ("half_max_flux", values[2], encircled_energy(half_max)),
                ("half_max_area", values[3], np.pi * half_max**2),
            ):
                airy_rows.append(
                    [pixel_lod, shift, metric, half_max, measured, analytic]
                )
    write_table(
        directory,
        "airy.csv",
        ["pixel_lod", "shift_px", "quantity", "radius_lod", "measured_10x", "analytic"],
        airy_rows,
    )

    coro = Coronagraph(fetch_yip("eac1_aavc_2d"))
    positions = list(perf._iter_xaxis_positions(coro))
    pixel_lod = float(coro.pixel_scale_arcsec.value)
    inside = [pos for pos in positions if perf._point_source_in_image(pos)]
    shipped = min(inside, key=lambda pos: abs(pos.separation - 20))
    convergence = []
    for name, psf, scale, ratio, center in (
        ("Airy", airy_psf(0.5, 0.3), 0.5, 0.5, (128, 128)),
        ("eac1_aavc_2d", shipped.psf, pixel_lod, 0.3, (shipped.py, shipped.px)),
    ):
        for factor in (5, 10, 20, 40):
            convergence.append(
                [name, factor, *measure(psf, scale, factor, ratio, center)]
            )
    quantities = ["circle_flux", "circle_area_lod2", "core_flux", "core_area_lod2"]
    write_table(
        directory, "convergence.csv", ["case", "oversample", *quantities], convergence
    )

    all_offsets = []
    for pos in inside:
        values = [
            measure(pos.psf, pixel_lod, factor, 0.3, (pos.py, pos.px))
            for factor in (10, 40)
        ]
        all_offsets.append([pos.separation, *values[0], *values[1]])
    write_table(
        directory,
        "shipped_gates.csv",
        [
            "offset_lod",
            *[q + "_10x" for q in quantities],
            *[q + "_40x" for q in quantities],
        ],
        all_offsets,
    )
    old = historical_performance("3fc8081")
    previous = historical_performance("436a876")
    offsets, old_flux = old.compute_throughput_curve(coro, oversample=2)
    _, new_flux = perf.compute_throughput_curve(coro)
    _, previous_flux = previous.compute_throughput_curve(coro)
    write_table(
        directory,
        "shipped_circle.csv",
        ["offset_lod", "old_cubic_2x", "previous_counted_10x", "fractional_10x"],
        zip(offsets, old_flux, previous_flux, new_flux, strict=True),
    )
    pos = positions[0]
    old_core = previous.truncation_core(pos.psf, 0.3, pixel_lod)
    new_core = measure(pos.psf, pixel_lod, 10, 0.3, (pos.py, pos.px))[2:]
    write_table(
        directory,
        "iwa_core.csv",
        ["method", "core_flux", "core_area_lod2"],
        [["previous_global", *map(float, old_core)], ["nominal_local", *new_core]],
    )
    print(
        f"Wrote tables to {directory}; shipped convergence offset {shipped.separation}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
