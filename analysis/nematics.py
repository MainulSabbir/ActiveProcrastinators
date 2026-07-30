"""
Nematics analysis branch for BARCODE.

Wraps the standalone scripts in the top-level ``Nematics/`` folder (imported
unchanged by adding that folder to ``sys.path``) and exposes a single
entry point, :func:`run_nematics_analysis`, that the processing pipeline
calls when the Nematics module is enabled.

Three image types (see :class:`core.config.NematicsConfig`):

* ``multipolarized`` - PolScope raw 2x2 polarization mosaic. Jones
  reconstruction -> Q tensor. Outputs: director-field overlay per image,
  correlation length (from the director), energy-spectrum plot + peak, and
  the length scale lambda* = 1/k*.
* ``microscopy`` - confocal/brightfield. ``biofilm_qtensor.run_pipeline``
  gives the structure-tensor Q field, director overlay and correlation
  length; same energy-spectrum outputs as above.
* ``sipli`` - Shear-Induced Polarized Light Imaging. Saturation min / max /
  average over the sample only (background excluded). No image output.

For a folder, the energy spectrum is TIME-AVERAGED over all frames (one
spectrum for the run), while a director overlay is written per image and
the correlation length is reported per image and averaged.
"""

import csv
import glob
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import numpy as np
from scipy import ndimage as ndi

from core import BarcodeConfig, InputConfig
from utils import vprint, set_verbose

# --- make the standalone Nematics scripts importable ---------------------
_NEMATICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Nematics"
)
if _NEMATICS_DIR not in sys.path:
    sys.path.insert(0, _NEMATICS_DIR)

import inspect as _inspect

import barcode_energy as be                 # energy spectrum E(k), peak k*
import biofilm_qtensor as bq                # microscopy structure-tensor pipeline
import polscope_qtensor_functions as pol    # PolScope reconstruction

# structure-tensor smoothing sigma that biofilm_qtensor.run_pipeline uses;
# reused to build the per-pixel microscopy energy-spectrum field so its
# director orientation matches the coarse overlay's convention exactly.
try:
    _STRUCT_SIGMA = _inspect.signature(bq.run_pipeline).parameters["struct_sigma"].default
except Exception:
    _STRUCT_SIGMA = 4

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# fixed number of radial k-bins so every frame's E(k) shares one k-grid
# (spectrum_setup's k-axis depends only on pixel_size, not image size, once
# nbins is fixed -> frames of different sizes can be averaged directly).
_SPECTRUM_NBINS = 50


# =========================================================================
# discovery + output helpers
# =========================================================================
def _discover_images(root_dir):
    """Return a sorted list of image files under root_dir (or [root_dir])."""
    if os.path.isfile(root_dir):
        return [root_dir]
    files = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in sorted(filenames):
            if name.startswith("._"):
                continue
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def _output_dir(root_dir):
    """Create and return the folder where Nematics results are written."""
    if os.path.isfile(root_dir):
        base = os.path.join(os.path.dirname(root_dir), "Nematics_Output")
    else:
        base = os.path.join(root_dir, "Nematics_Output")
    os.makedirs(base, exist_ok=True)
    return base


# =========================================================================
# Q-field helpers (shared by multipolarized + microscopy)
# =========================================================================
def _coarse_field_from_Q(Qxx, Qxy, window, overlap):
    """Block-average a per-pixel Q field into a grid compatible with
    ``biofilm_qtensor.plot_director_field`` and the correlation functions.

    Returns a dict with cx, cy, theta, S, Qxx, Qxy, step_x, step_y.
    """
    step = max(1, int(round(window * (1.0 - overlap))))
    ny, nx = Qxx.shape
    ys = np.arange(0, ny - window + 1, step)
    xs = np.arange(0, nx - window + 1, step)
    if len(ys) == 0:
        ys = np.array([0])
    if len(xs) == 0:
        xs = np.array([0])

    n_rows, n_cols = len(ys), len(xs)
    gxx = np.full((n_rows, n_cols), np.nan)
    gxy = np.full((n_rows, n_cols), np.nan)
    cx = np.zeros((n_rows, n_cols))
    cy = np.zeros((n_rows, n_cols))
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            block_xx = Qxx[y0:y0 + window, x0:x0 + window]
            block_xy = Qxy[y0:y0 + window, x0:x0 + window]
            gxx[i, j] = np.nanmean(block_xx)
            gxy[i, j] = np.nanmean(block_xy)
            cy[i, j] = y0 + window / 2.0
            cx[i, j] = x0 + window / 2.0

    theta = 0.5 * np.arctan2(gxy, gxx)
    S = np.sqrt(gxx ** 2 + gxy ** 2)
    return {
        "cx": cx, "cy": cy, "theta": theta, "S": S,
        "Qxx": gxx, "Qxy": gxy, "step_x": step, "step_y": step,
    }


def _correlation_data(field, image_shape):
    """Nematic correlation function + fit from a grid field, reusing the
    biofilm machinery. Returns a dict with xi_px, r2 and everything needed
    to draw the fit plot (r_vals, C_vals, pairs_vals, r_fit, C_fit,
    cutoff_used, r_max_cutoff, min_pairs_used), or None on failure.
    """
    try:
        valid = ~np.isnan(field["S"])
        if valid.sum() < 4:
            return None
        H, W = image_shape
        r_max = 0.5 * np.sqrt(H ** 2 + W ** 2)
        C2D, pair_counts, center = bq.nematic_correlation_2d(
            field["Qxx"], field["Qxy"], valid
        )
        r_vals, C_vals, pairs_vals, min_pairs_used = bq.radial_average_correlation(
            C2D, pair_counts, center,
            step_x=field["step_x"], step_y=field["step_y"], r_max=r_max,
        )
        xi_px, r2, r_fit, C_fit, cutoff_used = bq.fit_correlation_length(r_vals, C_vals)
        return {
            "xi_px": float(xi_px), "r2": float(r2),
            "r_vals": r_vals, "C_vals": C_vals, "pairs_vals": pairs_vals,
            "r_fit": r_fit, "C_fit": C_fit, "cutoff_used": cutoff_used,
            "r_max_cutoff": r_max, "min_pairs_used": min_pairs_used,
        }
    except Exception:
        traceback.print_exc()
        return None


def _save_director_overlay(background, field, out_png, title):
    """Save a director-field overlay PNG (rods colored by local order S)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    bq.plot_director_field(background, field, ax=ax)
    ax.set_title("")  # drop biofilm's built-in title (no titles on the plots)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_fine_director_overlay(background, theta, order, out_png, title,
                                target_rods=140):
    """High-resolution nematic director overlay for a PER-PIXEL field
    (e.g. PolScope, where the director is known at every pixel rather than
    on a coarse window grid).

    Samples the per-pixel director at a fine spacing (~`target_rods` rods
    across the short axis) and draws every rod as ONE LineCollection, so it
    stays fast even when dense. Rods are colored by local order (red = high,
    yellow = low), matching biofilm's plot_director_field convention.
    """
    H, W = background.shape
    step = max(4, int(round(min(H, W) / target_rods)))
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    X, Y = np.meshgrid(xs, ys)
    th = theta[Y, X]
    s = np.clip(order[Y, X], 0.0, 1.0)
    L = 0.9 * step
    dx = 0.5 * L * np.cos(th)
    dy = 0.5 * L * np.sin(th)
    p0 = np.stack([X - dx, Y - dy], axis=-1).reshape(-1, 2)
    p1 = np.stack([X + dx, Y + dy], axis=-1).reshape(-1, 2)
    segs = np.stack([p0, p1], axis=1)
    colors = cm.autumn(1.0 - s.ravel())

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(background, cmap="gray", origin="upper")
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=1.0,
                                     capstyle="round"))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_correlation_plot(cd, out_png, title):
    """Save the biofilm 3-panel nematic correlation-function fit plot."""
    try:
        fig, _axes = bq.plot_correlation_function(
            cd["r_vals"], cd["C_vals"], cd["pairs_vals"], cd["r_fit"], cd["C_fit"],
            cd["xi_px"], cd["cutoff_used"], cd["r_max_cutoff"], cd["min_pairs_used"],
        )
        for _a in np.ravel(_axes):   # drop all titles (no titles on the plots)
            _a.set_title("")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_png
    except Exception:
        traceback.print_exc()
        return None


# =========================================================================
# energy spectrum (time-averaged over frames)
# =========================================================================
def _accumulate_spectrum(frames, spec_pixel_size):
    """Time-average E(k) over a list of (Qxx, Qxy) frames.

    Returns (k_centers, E_avg, kstar) or (None, None, None) if empty.
    All frames share one k-grid because nbins and pixel_size are fixed.
    """
    if not frames:
        return None, None, None
    k_centers = None
    E_sum = None
    n = 0
    for Qxx, Qxy in frames:
        setup = be.spectrum_setup(Qxx.shape, spec_pixel_size, nbins=_SPECTRUM_NBINS)
        E = be.frame_energy_spectrum(Qxx, Qxy, setup)
        if k_centers is None:
            k_centers = setup["k_centers"]
            E_sum = np.zeros_like(E, dtype=float)
        E_sum += E
        n += 1
    E_avg = E_sum / max(n, 1)
    try:
        kstar = be.peak_wavenumber(k_centers, E_avg)
    except Exception:
        kstar = float("nan")
    return k_centers, E_avg, kstar


def _save_spectrum_plot(k_centers, E_avg, kstar, out_png, n_frames):
    """Plot the time-averaged energy spectrum E(k) with its peak marked."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_centers, E_avg, "-", color="steelblue", label="E(k)")
    lam = float("nan")
    if np.isfinite(kstar) and kstar > 0:
        lam = 1.0 / kstar
        ax.axvline(kstar, color="crimson", ls="--",
                   label=f"peak k* = {kstar:.3g}  (lambda* = {lam:.3g})")
    ax.set_xlabel(r"wavenumber $k = 1/\lambda$  ($\mu m^{-1}$)")
    ax.set_ylabel("elastic energy  E(k)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return lam


# =========================================================================
# per-type frame processing
# =========================================================================
def _process_multipolarized_frame(path, nem, out_dir):
    """PolScope frame -> director overlay + (Qxx, Qxy) + correlation length
    + average order parameter (via polscope_qtensor_functions.order_parameter)."""
    channels = pol.load_channels(path)
    alpha, delta = pol.reconstruct(channels)
    order_param = pol.order_parameter(channels)   # avg scalar order parameter
    Qxx = (delta / 2.0) * np.cos(2 * alpha)
    Qxy = (delta / 2.0) * np.sin(2 * alpha)

    # coarse grid only feeds the correlation-length metric (biofilm C(r)/fit).
    # Uses the RAW Q: coarse-graining already averages over window_size px, so
    # the pixel-scale noise the spectrum has to worry about is gone here.
    field = _coarse_field_from_Q(Qxx, Qxy, nem.window_size, nem.overlap)
    cd = _correlation_data(field, Qxx.shape)
    xi_px = cd["xi_px"] if cd else float("nan")
    r2 = cd["r2"] if cd else float("nan")

    # Band-limit Q for the ENERGY SPECTRUM only (see NematicsConfig.smoothing_sigma).
    # Without this the per-pixel Jones noise makes (grad Q)^2 climb to Nyquist and
    # the peak pins at ~2 px. The director overlay below still uses the raw
    # per-pixel field, so display resolution is unaffected.
    sigma = float(getattr(nem, "smoothing_sigma", 0.0) or 0.0)
    if sigma > 0:
        Qxx_spec = ndi.gaussian_filter(Qxx, sigma)
        Qxy_spec = ndi.gaussian_filter(Qxy, sigma)
    else:
        Qxx_spec, Qxy_spec = Qxx, Qxy

    I0, I45, I90, I135 = channels
    background = (I0 + I45 + I90 + I135) / 4.0
    # per-pixel order field for rod coloring (same quantity order_parameter averages)
    tot = (I0 + I45 + I90 + I135) / 2.0 + 1e-12
    S_pp = np.sqrt(((I0 - I90) / tot) ** 2 + ((I45 - I135) / tot) ** 2)
    stem = os.path.splitext(os.path.basename(path))[0]
    # high-resolution director overlay straight from the per-pixel PolScope field
    _save_fine_director_overlay(
        background, alpha, S_pp, os.path.join(out_dir, f"{stem}_director.png"),
        f"Director field - {stem}",
    )
    # NOTE: per user, multipolarized images do NOT get the correlation-function
    # plot (the xi metric is still reported); only microscopy saves that plot.
    return {"stem": stem, "Q": (Qxx_spec, Qxy_spec), "xi_px": xi_px, "r2": r2,
            "order_parameter": float(order_param), "spec_pixel_size": nem.pixel_size,
            "correlation_png": None}


def _per_pixel_Q_microscopy(corrected, mask, sigma=_STRUCT_SIGMA):
    """Full-resolution (per-pixel) Q field from biofilm's structure tensor,
    using the SAME orientation convention as coarse_grain_qtensor
    (director = gradient + pi/2, Q = 0.5 * coherence * [cos2t, sin2t]).

    Used only for the energy spectrum so it resolves finer scales than the
    coarse director grid (~window step). Two cleanups keep E(k) physical:

    * Background (outside the foreground mask) is zeroed so it injects no
      spurious energy.
    * Qxx/Qxy are band-limited by a Gaussian of width `sigma` (the same
      structure-tensor smoothing scale). Raw per-pixel director estimates
      from Sobel gradients are white-noise-like at the pixel scale, which
      makes (grad Q)^2 rise to the Nyquist frequency and pins the peak
      there. You cannot resolve director structure finer than the gradient
      smoothing length anyway, so this low-pass removes only aliased noise,
      leaving the physical energy-containing peak. Set sigma=0 to disable.
    """
    Jxx, Jyy, Jxy = bq.structure_tensor(corrected, sigma=sigma)
    trace = Jxx + Jyy
    with np.errstate(invalid="ignore", divide="ignore"):
        theta_dir = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
        coherence = np.where(trace > 1e-12,
                             np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / trace, 0.0)
    Qxx = 0.5 * coherence * np.cos(2 * theta_dir)
    Qxy = 0.5 * coherence * np.sin(2 * theta_dir)
    fg = mask.astype(bool)
    Qxx = np.where(fg, Qxx, 0.0)
    Qxy = np.where(fg, Qxy, 0.0)
    if sigma and sigma > 0:
        # NB: do NOT re-threshold with the mask afterwards -- a hard mask edge
        # is a step discontinuity that re-injects the very high-k energy this
        # low-pass removes. Let the background decay smoothly toward 0 instead.
        Qxx = ndi.gaussian_filter(Qxx, sigma)
        Qxy = ndi.gaussian_filter(Qxy, sigma)
    return Qxx.astype(np.float64), Qxy.astype(np.float64)


def _process_microscopy_frame(path, nem, out_dir):
    """Microscopy frame via biofilm structure-tensor pipeline.

    Director overlay + correlation length come from biofilm's coarse grid;
    the energy spectrum uses a full-resolution (per-pixel) Q field so it
    resolves finer scales than the coarse grid would.
    """
    res = bq.run_pipeline(
        path, window_size=nem.window_size, overlap=nem.overlap, show_plots=False,
    )
    field = res["field"]
    stem = os.path.splitext(os.path.basename(path))[0]
    _save_director_overlay(
        res["corrected"], field, os.path.join(out_dir, f"{stem}_director.png"),
        f"Director field - {stem}",
    )
    # correlation function + fit plot (same coarse grid biofilm used for xi)
    cd = _correlation_data(field, res["mask"].shape)
    xi_px = cd["xi_px"] if cd else float(res["xi_px"])
    r2 = cd["r2"] if cd else float(res.get("r2", float("nan")))
    corr_png = None
    if cd:
        corr_png = _save_correlation_plot(
            cd, os.path.join(out_dir, f"{stem}_correlation.png"),
            f"Nematic correlation - {stem}")
    # energy spectrum runs on the full-resolution Q field (spacing = 1 px),
    # so its physical pixel size is just um/px (not window-step scaled).
    Qxx, Qxy = _per_pixel_Q_microscopy(res["corrected"], res["mask"])
    spec_ps = nem.pixel_size
    # average order parameter is reported for MULTIPOLARIZED images only
    # (user request) -> microscopy returns None.
    return {"stem": stem, "Q": (Qxx, Qxy), "xi_px": xi_px, "r2": r2,
            "order_parameter": None, "spec_pixel_size": spec_ps,
            "correlation_png": corr_png}


def _process_sipli_frame(path, nem):
    """SIPLI frame -> saturation min/max/avg over the sample (no image)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _H, S, V = cv2.split(hsv)
    mask = V > nem.bg_threshold
    sample = S[mask].astype(np.float32)
    if sample.size == 0:
        raise ValueError("No sample pixels found (lower the background threshold).")
    return {
        "stem": os.path.splitext(os.path.basename(path))[0],
        "sat_min": float(sample.min()),
        "sat_max": float(sample.max()),
        "sat_avg": float(sample.mean()),
        "coverage_pct": float(100.0 * mask.mean()),
    }


# =========================================================================
# entry point
# =========================================================================
def run_nematics_analysis(root_dir: str, config: BarcodeConfig,
                          input_config: InputConfig):
    """Run the Nematics branch on a file or directory.

    Returns a summary dict ``{"title", "image_type", "output_dir", "text",
    "files", "metrics"}`` so the GUI can display the results, or None if
    nothing was processed. The same ``text`` is printed to the Processing
    Log and written to ``Nematics_Output/nematics_summary.txt``.
    """
    set_verbose(config.reader.verbose)
    nem = config.nematics_parameters
    image_type = (nem.image_type or "").lower()

    files = _discover_images(root_dir)
    if not files:
        print("Nematics: no image files found.")
        return None

    out_dir = _output_dir(root_dir)

    # emit() both prints (-> Processing Log) and records the summary text
    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    def finish(title, metrics, director_images=None, spectrum_image=None,
               correlation_images=None):
        text = "\n".join(summary_lines)
        try:
            with open(os.path.join(out_dir, "nematics_summary.txt"), "w") as fh:
                fh.write(text + "\n")
        except Exception as e:
            print(f"  (could not write summary txt: {e})")
        return {"title": title, "image_type": image_type, "output_dir": out_dir,
                "text": text, "files": files, "metrics": metrics,
                "director_images": director_images or [],      # [(stem, png_path), ...]
                "correlation_images": correlation_images or [],  # [(stem, png_path), ...]
                "spectrum_image": spectrum_image}              # png_path or None

    emit(f"Nematics analysis - {image_type}")
    emit(f"{len(files)} image(s)")
    emit(f"Output folder: {out_dir}")
    emit("")

    # ----- SIPLI: saturation only, no image output -----
    if image_type == "sipli":
        rows = []
        for f in files:
            try:
                rows.append(_process_sipli_frame(f, nem))
            except Exception as e:
                emit(f"  {os.path.basename(f)}: {e}")
        if not rows:
            emit("SIPLI: no results.")
            return finish("Nematics SIPLI - Saturation", {})
        csv_path = os.path.join(out_dir, "sipli_saturation.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "sat_min", "sat_max", "sat_avg", "coverage_pct"])
            for r in rows:
                w.writerow([r["stem"], f"{r['sat_min']:.3f}", f"{r['sat_max']:.3f}",
                            f"{r['sat_avg']:.3f}", f"{r['coverage_pct']:.2f}"])
        smin = min(r["sat_min"] for r in rows)
        smax = max(r["sat_max"] for r in rows)
        savg = float(np.mean([r["sat_avg"] for r in rows]))
        emit("Saturation over the sample (0-255, background excluded):")
        emit(f"  min      {smin:.1f}   ({smin / 255 * 100:.1f}%)")
        emit(f"  max      {smax:.1f}   ({smax / 255 * 100:.1f}%)")
        emit(f"  average  {savg:.1f}   ({savg / 255 * 100:.1f}%)")
        if len(rows) > 1:
            emit(f"  (min/max across {len(rows)} images; average = mean of per-image averages)")
        emit("")
        emit(f"Saved: {os.path.basename(csv_path)}")
        return finish("Nematics SIPLI - Saturation",
                      {"sat_min": smin, "sat_max": smax, "sat_avg": savg,
                       "n_images": len(rows)})

    # ----- multipolarized / microscopy -----
    if image_type == "multipolarized":
        process = _process_multipolarized_frame
    elif image_type == "microscopy":
        process = _process_microscopy_frame
    else:
        emit(f"Unknown image type '{nem.image_type}'.")
        return finish("Nematics - Error", {})

    # average order parameter is reported for MULTIPOLARIZED images only (user request)
    report_op = (image_type == "multipolarized")

    frames = []          # list of (Qxx, Qxy) for the spectrum
    spec_pixel_size = None
    metric_rows = []     # per-image (stem, xi_px, xi_um, r2, order_param)
    director_images = []     # [(stem, director_png_path), ...] for the GUI viewer
    correlation_images = []  # [(stem, correlation_png_path), ...]
    emit("Per-image results:")
    for f in files:
        try:
            r = process(f, nem, out_dir)
        except Exception as e:
            emit(f"  {os.path.basename(f)}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        frames.append(r["Q"])
        spec_pixel_size = r["spec_pixel_size"]
        director_images.append(
            (r["stem"], os.path.join(out_dir, f"{r['stem']}_director.png")))
        if r.get("correlation_png"):
            correlation_images.append((r["stem"], r["correlation_png"]))
        xi_um = r["xi_px"] * nem.pixel_size if np.isfinite(r["xi_px"]) else float("nan")
        op = r["order_parameter"]
        metric_rows.append((r["stem"], r["xi_px"], xi_um, r["r2"], op))
        if report_op:
            emit(f"  {r['stem']}:  order param = {op:.3f},  "
                 f"correlation length = {r['xi_px']:.1f} px ({xi_um:.3g} um),  R^2 = {r['r2']:.3f}")
        else:
            emit(f"  {r['stem']}:  correlation length = {r['xi_px']:.1f} px "
                 f"({xi_um:.3g} um),  R^2 = {r['r2']:.3f}")

    if not frames:
        emit("No frames processed successfully.")
        return finish("Nematics - No results", {})

    # time-averaged energy spectrum + peak / length scale
    k_centers, E_avg, kstar = _accumulate_spectrum(frames, spec_pixel_size)
    length_scale = float("nan")
    spectrum_png = None
    if k_centers is not None:
        spectrum_png = os.path.join(out_dir, "energy_spectrum.png")
        length_scale = _save_spectrum_plot(k_centers, E_avg, kstar, spectrum_png, len(frames))
        np.savez(os.path.join(out_dir, "energy_spectrum.npz"),
                 k=k_centers, E=E_avg, kstar=kstar)

    # run-level averages
    op_vals = [o for _s, _p, _u, _r, o in metric_rows
               if o is not None and np.isfinite(o)]
    xi_um_vals = [x for _s, _p, x, _r, _o in metric_rows if np.isfinite(x)]
    mean_op = float(np.mean(op_vals)) if op_vals else float("nan")
    mean_xi_um = float(np.mean(xi_um_vals)) if xi_um_vals else float("nan")

    emit("")
    emit("Summary (averaged over all images):")
    if report_op:
        emit(f"  average order parameter    = {mean_op:.4f}")
    emit(f"  mean correlation length    = {mean_xi_um:.4g} um")
    emit(f"  energy-spectrum peak k*    = {kstar:.4g} um^-1")
    emit(f"  length scale (lambda*=1/k*) = {length_scale:.4g} um")
    emit("")

    # summary CSV of per-image metrics + run-level values
    csv_path = os.path.join(out_dir, "nematics_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        if report_op:
            w.writerow(["image", "order_parameter", "correlation_length_px",
                        "correlation_length_um", "fit_r2"])
            for stem, xi_px, xi_um, r2, op in metric_rows:
                w.writerow([stem, f"{op:.4f}", f"{xi_px:.3f}", f"{xi_um:.4g}", f"{r2:.3f}"])
        else:
            w.writerow(["image", "correlation_length_px", "correlation_length_um", "fit_r2"])
            for stem, xi_px, xi_um, r2, _op in metric_rows:
                w.writerow([stem, f"{xi_px:.3f}", f"{xi_um:.4g}", f"{r2:.3f}"])
        w.writerow([])
        if report_op:
            w.writerow(["average_order_parameter", f"{mean_op:.4f}"])
        w.writerow(["mean_correlation_length_um", f"{mean_xi_um:.4g}"])
        w.writerow(["energy_peak_k_um^-1", f"{kstar:.5g}"])
        w.writerow(["energy_length_scale_um", f"{length_scale:.5g}"])
    saved = [os.path.basename(csv_path), "energy_spectrum.png", "per-image *_director.png"]
    if correlation_images:
        saved.append("*_correlation.png")
    emit(f"Saved: {', '.join(saved)}")

    metrics = {"mean_correlation_length_um": mean_xi_um,
               "energy_peak_k": float(kstar), "length_scale_um": length_scale,
               "n_images": len(frames)}
    if report_op:
        metrics["average_order_parameter"] = mean_op

    return finish(
        f"Nematics {image_type.capitalize()} - Results",
        metrics,
        director_images=director_images,
        spectrum_image=spectrum_png,
        correlation_images=correlation_images,
    )
