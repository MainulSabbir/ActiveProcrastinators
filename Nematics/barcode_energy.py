"""
Shared helpers for the BARCODE elastic-energy pipeline.

This is the ONE place where the file conventions and the physics live; every
other script imports from here. Works on ANY processed folder that has a
`Q_data/` directory of `*_Q.npz` files (Qxx, Qxy), regardless of frame numbering
(pic--1..., pic--1001...) or image size.

DATA FLOW (what happens to the Q tensor)
----------------------------------------
    Q_data/pic--N_Q.npz                     one file per movie frame
        |  load_Q()                          read the two arrays Qxx, Qxy
        v
    Qxx, Qxy                                 BARCODE stores Qxx = S*cos2theta,
        |                                                    Qxy = S*sin2theta
        |  energy_maps()  -- the ADAPTER + physics
        |    theta = 0.5*atan2(Qxy, Qxx)      director angle
        |    S     = sqrt(Qxx^2 + Qxy^2)      scalar order parameter
        |    Q(2x2) = build_Q(theta, S)       full tensor (elastic_energy_2d.py)
        |    E_total,E_L1,E_L2,E_L6 = elastic_energy_2d(Q, ...)
        v
    per-pixel energy maps  ->  plot_energy_map.py  / plot_energy_hist.py

    ...or, for the spectrum, the Q field goes straight to Fourier space:
        frame_energy_spectrum() -> E(k) per frame
        timeavg_energy_spectrum() -> averaged E(k) -> plot_energy_spectrum.py

GLOSSARY
--------
    theta   director orientation angle (radians)
    S       order parameter / coherency (0 = defect core, ~1 = well aligned)
    E_total elastic energy density from Eq. 2 (units of the L constants)
    E(k)    elastic energy per wavenumber shell; peak k* = dominant scale
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elastic_energy_2d as ee

Q_GLOB = "*_Q.npz"          # naming BARCODE uses for its Q output
Q_KEYS = ("Qxx", "Qxy")     # array names inside each npz


def frame_index(path):
    """Extract the integer frame number from a `..--<N>_Q.npz` filename."""
    m = re.search(r"--(\d+)_Q\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_q_files(processed_dir):
    """All Q files in <processed_dir>/Q_data, sorted by frame number."""
    q_dir = os.path.join(processed_dir, "Q_data")
    files = sorted(glob.glob(os.path.join(q_dir, Q_GLOB)), key=frame_index)
    if not files:
        raise SystemExit(f"no {Q_GLOB} files found in {q_dir}")
    return files


def frame_range(processed_dir):
    """(first, last, count) frame numbers available -- handy for error messages."""
    files = list_q_files(processed_dir)
    return frame_index(files[0]), frame_index(files[-1]), len(files)


def resolve_frame(processed_dir, frame):
    """Return the Q-file path for `frame`.

    `frame` is matched against the ACTUAL frame numbers in the folder, so it
    works whether a dataset starts at 1 or 1001. frame=None picks the middle
    frame. An out-of-range request fails with the available range, not a
    confusing FileNotFoundError.
    """
    files = list_q_files(processed_dir)
    by_num = {frame_index(f): f for f in files}
    if frame is None:
        return files[len(files) // 2]
    if frame in by_num:
        return by_num[frame]
    lo, hi, n = frame_range(processed_dir)
    raise SystemExit(f"frame {frame} not in {processed_dir}: "
                     f"available {lo}..{hi} ({n} frames)")


def load_Q(path):
    """Read (Qxx, Qxy) as float64 from a BARCODE Q npz."""
    with np.load(path) as d:
        return d[Q_KEYS[0]].astype(np.float64), d[Q_KEYS[1]].astype(np.float64)


def energy_maps(Qxx, Qxy, k11, k33, s0, core_threshold=0.15, pixel_size=1.0):
    """Elastic energy maps for one frame -- the adapter from BARCODE Q to Eq. 2.

    k11, k33  Frank splay/bend constants; s0 the reference order parameter used
              only to turn K -> L. core_threshold flags low-S defect cores.
    Returns a dict of per-pixel maps plus the L constants that were used.
    """
    # 1) recover director angle and order parameter from the stored components
    theta = 0.5 * np.arctan2(Qxy, Qxx)                  # Qxx=S cos2t, Qxy=S sin2t
    S = np.sqrt(Qxx ** 2 + Qxy ** 2)                    # = op_* columns of BARCODE

    # 2) rebuild the full traceless 2x2 tensor with the paper's own routine
    Q = ee.build_Q(theta, S, trace=2)                  # trace=2 = BARCODE convention

    # 3) convert Frank constants to Landau-de Gennes L's, then evaluate Eq. 2
    L1, L2, L6 = ee.L_from_K(k11, k33, s0, trace=2)
    Etot, E1, E2, E6 = ee.elastic_energy_2d(Q, pixel_size, pixel_size, L1, L2, L6)

    # 4) mark defect cores (Frank energy is not meaningful there)
    core = S < core_threshold * float(np.median(S))
    return {
        "E_total": Etot, "E_L1": E1, "E_L2": E2, "E_L6": E6,
        "S": S, "theta": theta, "core": core, "L": (L1, L2, L6),
    }


# --------------------------------------------------------------------------
# elastic energy spectrum  (Fig. 3c of Sokolov et al., Adv. Mater. 2025)
# --------------------------------------------------------------------------
#
# One-constant elastic energy density is (grad Q)^2, so in Fourier space each
# mode contributes (grad energy) |Q_hat|^2, and the total elastic energy splits
# by wavenumber as
#       E(k) = sum_{|q| in shell k}  (2*pi*k)^2 (|Qxx_hat|^2 + |Qxy_hat|^2)
# with sum_k E(k) = total elastic energy. The peak k* is the energy-containing
# scale, lambda* = 1/k*.
#
# UNITS: the axis wavenumber k is the CYCLIC wavenumber k = 1/lambda, in um^-1,
# matching the article (its S(k) axes are labelled um^-1). The physical energy
# weight per mode still uses the angular gradient factor (2*pi*k)^2 -- that is
# just a constant on the shape, so it does not move the peak, but it keeps E(k)
# a genuine elastic energy. Do NOT confuse this k with the angular 2*pi/lambda.

def spectrum_setup(shape, pixel_size, nbins=None):
    """Precompute the k-grid, taper and bin assignment once for a dataset.

    Doing this per frame (via np.histogram) is the slow path; precomputing the
    radial bin index and reusing np.bincount per frame is ~50x faster.
    """
    ny, nx = shape
    window = np.outer(np.hanning(ny), np.hanning(nx))       # Hann taper: kills edge leakage
    fx = np.fft.fftfreq(nx, d=pixel_size)                   # cyclic wavenumber, cycles/length
    fy = np.fft.fftfreq(ny, d=pixel_size)                   # = 1/lambda, the article's k (um^-1)
    FX, FY = np.meshgrid(fx, fy)
    F = np.sqrt(FX ** 2 + FY ** 2)                          # |k| = 1/lambda per mode

    K2 = (2 * np.pi * F) ** 2                               # (grad Q)^2 weight = (angular k)^2
    f_nyq = 1.0 / (2 * pixel_size)                          # Nyquist, cycles/length
    nbins = nbins or (min(ny, nx) // 2)
    fbins = np.linspace(0, f_nyq, nbins + 1)
    idx = np.digitize(F.ravel(), fbins) - 1                 # which shell each mode falls in
    valid = (idx >= 0) & (idx < nbins)
    return {
        "window": window, "K2": K2.ravel(),
        "idx": idx[valid], "valid": valid, "nbins": nbins,
        "k_centers": 0.5 * (fbins[:-1] + fbins[1:]),        # cyclic k (1/lambda), um^-1
    }


def frame_energy_spectrum(Qxx, Qxy, setup):
    """Elastic energy per radial k-shell for one frame (unnormalised)."""
    w = setup["window"]
    Fx = np.fft.fft2((Qxx - Qxx.mean()) * w)               # subtract k=0, then taper
    Fy = np.fft.fft2((Qxy - Qxy.mean()) * w)
    weight = setup["K2"] * (np.abs(Fx) ** 2 + np.abs(Fy) ** 2).ravel()
    return np.bincount(setup["idx"], weights=weight[setup["valid"]],
                       minlength=setup["nbins"])[: setup["nbins"]]


def peak_wavenumber(k_centers, E):
    """Peak of E(k) with parabolic sub-bin refinement; skips the DC end."""
    i0 = 2                                                  # ignore lowest bins (residual DC/leakage)
    i = i0 + int(np.argmax(E[i0:]))
    denom = E[i - 1] - 2 * E[i] + E[i + 1] if 0 < i < len(E) - 1 else 0.0
    shift = 0.5 * (E[i - 1] - E[i + 1]) / denom if denom != 0 else 0.0
    dk = k_centers[1] - k_centers[0]
    return k_centers[i] + shift * dk


def timeavg_energy_spectrum(processed_dir, pixel_size, fmin=None, fmax=None, nbins=None):
    """Time-averaged E(k) over a dataset. Returns (k_centers, E_avg, kstar, files)."""
    files = list_q_files(processed_dir)
    if fmin is not None:
        files = [f for f in files if frame_index(f) >= fmin]
    if fmax is not None:
        files = [f for f in files if frame_index(f) <= fmax]
    if not files:
        raise SystemExit("no frames in the requested range")

    setup = spectrum_setup(load_Q(files[0])[0].shape, pixel_size, nbins)
    E = np.zeros(setup["nbins"])
    for f in files:
        E += frame_energy_spectrum(*load_Q(f), setup)
    E /= len(files)
    return setup["k_centers"], E, peak_wavenumber(setup["k_centers"], E), files
