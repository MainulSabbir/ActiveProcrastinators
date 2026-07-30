"""
biofilm_qtensor.py
===================

Director field / Q-tensor extraction and nematic correlation-length
analysis for confocal biofilm images.

Pipeline: grayscale -> illumination correction -> foreground mask ->
structure-tensor edge detection -> windowed coarse-graining -> director
field + Q-tensor -> nematic correlation function C(r) -> correlation
length xi.

Usage in a notebook
--------------------
    import biofilm_qtensor as bq

    results = bq.run_pipeline(
        "/path/to/image.tif",
        window_size=32, overlap=0.5,
        save_prefix="biofilm",     # writes biofilm_Q.npz + biofilm_results.npz
    )

    Qxx = results["field"]["Qxx"]
    Qxy = results["field"]["Qxy"]
    S   = results["field"]["S"]
    xi_px = results["xi_px"]

All the individual functions below (load_grayscale, correct_illumination,
make_mask, structure_tensor, coarse_grain_qtensor, plot_director_field,
plot_heatmap, nematic_correlation_2d, radial_average_correlation,
fit_correlation_length, save_qtensor_npz) are also importable directly if
you want to customize a step instead of using run_pipeline() end-to-end.
"""

import numpy as np
import cv2
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm


# ==========================================================================
# Step 1: load image, correct illumination, build foreground mask
# ==========================================================================

def load_grayscale(path, channel="auto"):
    """Load image and return a float grayscale array in [0, 1] where
    bacteria (bright signal) -> high values, background -> ~0."""
    raw = tifffile.imread(path)  # RGB(A) channel order, reliable for TIFFs
    if raw.ndim == 2:
        gray = raw.astype(np.float64)
    else:
        chans = [raw[..., i].astype(np.float64) for i in range(raw.shape[2])]
        if raw.shape[2] >= 4:
            chans = chans[:3]  # drop alpha
        if channel == "auto":
            variances = [c.var() for c in chans]
            gray = chans[int(np.argmax(variances))]
        elif channel == "gray":
            gray = np.mean(chans, axis=0)
        else:
            idx = {"R": 0, "G": 1, "B": 2}[channel]
            gray = chans[idx]
    gray = gray - gray.min()
    if gray.max() > 0:
        gray = gray / gray.max()
    return gray


def correct_illumination(gray, sigma):
    """Estimate the slowly-varying illumination field with a large Gaussian
    blur and divide it out."""
    background = ndi.gaussian_filter(gray, sigma=sigma)
    background = np.clip(background, 1e-3, None)
    corrected = gray / background
    corrected = corrected / corrected.max()
    return corrected


def make_mask(corrected, method="adaptive", block=51, C=-2, min_size=20,
              smooth_sigma=0, open_kernel=0):
    """Foreground mask (bacteria = 1, background = 0)."""
    img8 = (corrected * 255).clip(0, 255).astype(np.uint8)
    if smooth_sigma and smooth_sigma > 0:
        img8 = cv2.GaussianBlur(img8, (0, 0), sigmaX=smooth_sigma)
    if method == "adaptive":
        block = block if block % 2 == 1 else block + 1
        mask = cv2.adaptiveThreshold(
            img8, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=block, C=C
        )
    else:  # global otsu
        _, mask = cv2.threshold(img8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)

    if open_kernel and open_kernel > 0:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean[labels == i] = 1
    return clean


# ==========================================================================
# Step 2: structure tensor + windowed coarse-graining -> director / Q-tensor
# ==========================================================================

def structure_tensor(gray, sigma):
    """Sobel gradients (edge detection) -> local structure tensor J,
    smoothed over `sigma`."""
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    Jxx = ndi.gaussian_filter(Gx * Gx, sigma)
    Jyy = ndi.gaussian_filter(Gy * Gy, sigma)
    Jxy = ndi.gaussian_filter(Gx * Gy, sigma)
    return Jxx, Jyy, Jxy


def window_starts(length, window_size, overlap):
    step = max(1, int(round(window_size * (1 - overlap))))
    starts = list(range(0, max(length - window_size, 0) + 1, step))
    if not starts:
        starts = [0]
    if starts[-1] + window_size < length:
        starts.append(length - window_size)
    return starts


def coarse_grain_qtensor(Jxx, Jyy, Jxy, mask, window_size, overlap, min_frac):
    """Average the structure tensor over each window (restricted to
    foreground pixels) and convert to director angle / order parameter /
    Q-tensor. Windows without enough foreground are left as NaN.

    Returns a dict: cx, cy (window-center pixel coords), theta, S, Qxx, Qxy
    (all shape (n_rows, n_cols)), plus window_size, step_x, step_y.
    """
    H, W = mask.shape
    ys = window_starts(H, window_size, overlap)
    xs = window_starts(W, window_size, overlap)
    ny, nx = len(ys), len(xs)

    cy = np.zeros((ny, nx))
    cx = np.zeros((ny, nx))
    theta = np.full((ny, nx), np.nan)
    S = np.full((ny, nx), np.nan)
    Qxx = np.full((ny, nx), np.nan)
    Qxy = np.full((ny, nx), np.nan)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            cy[i, j] = y + window_size / 2
            cx[i, j] = x + window_size / 2

            sub_mask = mask[y:y + window_size, x:x + window_size]
            n_fg = sub_mask.sum()
            if n_fg == 0 or (n_fg / sub_mask.size) < min_frac:
                continue

            jxx = (Jxx[y:y + window_size, x:x + window_size] * sub_mask).sum() / n_fg
            jyy = (Jyy[y:y + window_size, x:x + window_size] * sub_mask).sum() / n_fg
            jxy = (Jxy[y:y + window_size, x:x + window_size] * sub_mask).sum() / n_fg

            trace = jxx + jyy
            if trace < 1e-12:
                continue

            theta_grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
            theta_dir = theta_grad + np.pi / 2
            coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / trace

            theta[i, j] = theta_dir
            S[i, j] = coherence
            Qxx[i, j] = 0.5 * coherence * np.cos(2 * theta_dir)
            Qxy[i, j] = 0.5 * coherence * np.sin(2 * theta_dir)

    return {"cx": cx, "cy": cy, "theta": theta, "S": S, "Qxx": Qxx, "Qxy": Qxy,
            "window_size": window_size,
            "step_x": xs[1] - xs[0] if len(xs) > 1 else window_size,
            "step_y": ys[1] - ys[0] if len(ys) > 1 else window_size}


# ==========================================================================
# Plotting
# ==========================================================================

def plot_director_field(gray_bg, field, rod_length_scale=0.9, ax=None):
    """Director field overlay: short rods colored by local order S."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(gray_bg, cmap="gray", origin="upper")
    cx, cy, theta, S = field["cx"], field["cy"], field["theta"], field["S"]
    step = min(field["step_x"], field["step_y"])
    L = rod_length_scale * step
    for i in range(cx.shape[0]):
        for j in range(cx.shape[1]):
            if np.isnan(theta[i, j]):
                continue
            th = theta[i, j]
            s = S[i, j]
            dx = 0.5 * L * np.cos(th)
            dy = 0.5 * L * np.sin(th)
            x0, y0 = cx[i, j], cy[i, j]
            ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy],
                    color=cm.autumn(1 - s), linewidth=1.6, solid_capstyle="round")
    ax.set_xlim(0, gray_bg.shape[1]); ax.set_ylim(gray_bg.shape[0], 0)
    ax.set_title("Director field (color = local order S: red high, yellow low)")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_heatmap(data, title, ax=None, cmap="viridis", vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# ==========================================================================
# Step 3: nematic correlation function + correlation length
# ==========================================================================

def nematic_correlation_2d(Qxx, Qxy, valid):
    """2D nematic autocorrelation of the coarse-grained Q-tensor grid,
    via FFT (Wiener-Khinchin), normalized by the number of valid pairs
    contributing at each lag so that NaN/skipped windows don't bias C(r).

    IMPORTANT: we correlate the *fluctuation* of Q around its mean, not Q
    itself. If the biofilm has any net global alignment (a bulk preferred
    direction), the raw <Q(0)Q(r)> plateaus at <Q>^2 instead of decaying to
    0 -- that plateau reflects genuine long-range order, not local
    correlation length, and would otherwise silently inflate xi.

    Returns (C_norm, pair_counts, center_index) with C_norm[center] == 1.
    """
    ny, nx = Qxx.shape
    mean_xx = Qxx[valid].mean()
    mean_xy = Qxy[valid].mean()
    dQxx = np.where(valid, Qxx - mean_xx, 0.0)
    dQxy = np.where(valid, Qxy - mean_xy, 0.0)
    w = valid.astype(float)

    pad_y, pad_x = 2 * ny, 2 * nx  # zero-pad to avoid circular wrap-around

    def autocorr(a):
        F = np.fft.fft2(a, s=(pad_y, pad_x))
        c = np.fft.ifft2(F * np.conj(F)).real
        return np.fft.fftshift(c)

    C_raw = autocorr(dQxx) + autocorr(dQxy)
    pair_counts = autocorr(w)

    with np.errstate(invalid="ignore", divide="ignore"):
        C_norm = np.where(pair_counts > 0.5, C_raw / pair_counts, np.nan)

    center = (pad_y // 2, pad_x // 2)
    C_norm = C_norm / C_norm[center]
    return C_norm, pair_counts, center


def radial_average_correlation(C_norm, pair_counts, center, step_x, step_y,
                                min_pairs_fraction=0.02, r_max=None):
    """Bin the 2D correlation map by radial distance |r|.

    Two guards against the noisy/artifactual far tail:
      - `r_max`: hard cutoff on distance (pass ~half the image diagonal --
        past that, so few window pairs are that far apart that one or two
        of them can swing the whole bin, e.g. near-corner-to-corner pairs).
      - `min_pairs_fraction`: a bin is only kept if it's backed by at least
        this fraction of the total valid-window count (pair_counts at r=0)
        -- an adaptive version of a fixed min-pairs threshold that scales
        with how much data you actually have.

    Returns r_centers, C_binned, and pairs_binned (mean pair count per bin,
    for diagnosing exactly where the statistics run out), and min_pairs used.
    """
    pad_y, pad_x = C_norm.shape
    yy, xx = np.indices((pad_y, pad_x))
    dy = (yy - center[0]) * step_y
    dx = (xx - center[1]) * step_x
    r = np.sqrt(dx ** 2 + dy ** 2)

    n_valid_windows = pair_counts[center]  # pair count at r=0 = total valid windows
    min_pairs = max(5, min_pairs_fraction * n_valid_windows)

    ok = np.isfinite(C_norm) & (pair_counts >= min_pairs)
    if r_max is not None:
        ok &= (r <= r_max)
    r_ok, C_ok, pairs_ok = r[ok], C_norm[ok], pair_counts[ok]

    r_bin_max = r_ok.max()
    n_bins = max(10, int(r_bin_max / min(step_x, step_y)))
    bins = np.linspace(0, r_bin_max, n_bins + 1)
    bin_idx = np.digitize(r_ok, bins) - 1

    r_centers = 0.5 * (bins[:-1] + bins[1:])
    C_binned = np.full(n_bins, np.nan)
    pairs_binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.any():
            C_binned[b] = C_ok[sel].mean()
            pairs_binned[b] = pairs_ok[sel].mean()

    keep = np.isfinite(C_binned)
    return r_centers[keep], C_binned[keep], pairs_binned[keep], min_pairs


def fit_correlation_length(r, C, decay_threshold=0.12, min_points=4):
    """Fit C(r) = A * exp(-r/xi) to just the reliable near-field decay.

    Two deliberate choices, from earlier debugging on real data:
      1. The fit window stops automatically at the first r where C drops
         below `decay_threshold` (default 0.12) -- past that point C is
         small enough that noise/far-tail artifacts dominate and
         shouldn't be allowed to drag the fit around.
      2. The fit itself is nonlinear least squares directly on C(r)
         (scipy.optimize.curve_fit), NOT on log(C). A log-linear fit
         weights every point equally regardless of how noisy it is in
         relative terms -- that let near-zero far-tail points drag the
         fitted decay rate down (and xi up) in an earlier version.

    Returns xi, R^2 (computed on the raw C(r) fit, not log-space), a dense
    curve for plotting, and the cutoff radius that was actually used.
    """
    r, C = np.asarray(r), np.asarray(C)
    pos = r > 0
    r_pos, C_pos = r[pos], C[pos]

    below = np.where(C_pos < decay_threshold)[0]
    cutoff = r_pos[below[0]] if len(below) else r_pos.max()

    sel = r_pos <= cutoff
    if sel.sum() < min_points:
        order = np.argsort(r_pos)
        sel = np.zeros_like(r_pos, dtype=bool)
        sel[order[:min_points]] = True
        cutoff = r_pos[sel].max()

    r_fit, C_fit_data = r_pos[sel], C_pos[sel]

    def model(rr, A, xi):
        return A * np.exp(-rr / xi)

    xi_guess = max(r_fit[-1] / 2, 1.0)
    (A_fit, xi), _ = curve_fit(
        model, r_fit, C_fit_data, p0=[max(C_fit_data[0], 1e-3), xi_guess], maxfev=5000
    )

    pred = model(r_fit, A_fit, xi)
    ss_res = np.sum((C_fit_data - pred) ** 2)
    ss_tot = np.sum((C_fit_data - C_fit_data.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    r_dense = np.linspace(0, r_fit.max(), 100)
    C_dense = model(r_dense, A_fit, xi)
    return xi, r2, r_dense, C_dense, cutoff


def plot_correlation_function(r_vals, C_vals, pairs_vals, r_fit, C_fit,
                               xi_px, cutoff_used, r_max_cutoff, min_pairs_used):
    """The 3-panel C(r) / log-linear-check / pairs-per-bin diagnostic plot."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(r_vals, C_vals, "o", ms=4, label="C(r) data")
    axes[0].plot(r_fit, C_fit, "-", color="crimson", label=f"exp fit, xi={xi_px:.1f}px")
    axes[0].axvline(cutoff_used, color="gray", ls="--", lw=1, label="fit cutoff")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("r (px)"); axes[0].set_ylabel("C(r)")
    axes[0].set_title("Nematic correlation function"); axes[0].legend(fontsize=8)

    pos = C_vals > 0
    axes[1].semilogy(r_vals[pos], C_vals[pos], "o", ms=4)
    axes[1].semilogy(r_fit, C_fit, "-", color="crimson")
    axes[1].axvline(cutoff_used, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("r (px)"); axes[1].set_ylabel("C(r)  (log scale)")
    axes[1].set_title("Log-linear check (should look straight if exponential)")

    axes[2].plot(r_vals, pairs_vals, "o", ms=4, color="teal")
    axes[2].axhline(min_pairs_used, color="gray", ls="--", lw=1, label="min pairs/bin")
    axes[2].axvline(r_max_cutoff, color="crimson", ls="--", lw=1, label="hard r cutoff")
    axes[2].set_xlabel("r (px)"); axes[2].set_ylabel("mean pairs per bin")
    axes[2].set_title("Statistics: where do pairs run out?"); axes[2].legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


# ==========================================================================
# Saving
# ==========================================================================

def save_qtensor_npz(Qxx, Qxy, path):
    """Save the two independent Q-tensor components as float32, matching
    the storage convention:
        Qyy = -Qxx  (traceless condition -- not stored separately)
        load via: data = np.load(path); Qxx = data['Qxx']; Qxy = data['Qxy']
    """
    np.savez(path, Qxx=Qxx.astype(np.float32), Qxy=Qxy.astype(np.float32))
    print(f"Saved Q-tensor ({Qxx.shape[0]}x{Qxx.shape[1]}, float32) to {path}")


# ==========================================================================
# High-level convenience wrapper: image path in, everything out
# ==========================================================================

def run_pipeline(image_path,
                  channel="auto",
                  illum_sigma=100,
                  mask_method="adaptive", mask_smooth_sigma=0.5,
                  adaptive_block=31, adaptive_c=2,
                  min_object_size=10, open_kernel=0,
                  struct_sigma=4,
                  window_size=32, overlap=0.5, min_foreground_fraction=0.05,
                  rod_length_scale=0.9,
                  max_r_fraction_of_diagonal=0.5, min_pairs_fraction=0.02,
                  decay_threshold=0.12,
                  save_prefix=None,
                  show_plots=True):
    """Run the full pipeline on one image: load -> illumination-correct ->
    mask -> structure tensor -> Q-tensor -> director field plot -> Q-tensor
    heatmaps -> nematic correlation function -> correlation length.

    Parameters mirror the notebook's parameters cell -- see the module
    docstring / individual function docstrings for what each one does.

    save_prefix : str or None
        If given, writes two files: f"{save_prefix}_Q.npz" (Qxx, Qxy only,
        float32 -- see save_qtensor_npz) and f"{save_prefix}_results.npz"
        (theta, S, Qxx, Qxy, cx, cy -- the full field). If None, nothing
        is saved to disk (call save_qtensor_npz yourself if you want it
        later, using results["field"]["Qxx"/"Qxy"]).
    show_plots : bool
        If True (default), displays the mask/correction panel, director
        field overlay, Q-tensor heatmaps, and correlation function plot.
        Set False for batch-processing many images without popping up
        figures for each one.

    Returns
    -------
    dict with keys:
        gray, corrected, mask   : the three preprocessing stages
        field                   : dict with cx, cy, theta, S, Qxx, Qxy,
                                   window_size, step_x, step_y
        xi_px, r2, cutoff_used  : correlation-length fit results
        r_vals, C_vals, pairs_vals : the binned correlation function data
        figures                 : dict of the matplotlib Figures created
                                   (keys: "preprocessing", "director_field",
                                   "qtensor_heatmaps", "correlation")
    """
    figures = {}

    # --- Step 1: load, correct illumination, mask ---
    gray = load_grayscale(image_path, channel=channel)
    corrected = correct_illumination(gray, sigma=illum_sigma)
    mask = make_mask(corrected, method=mask_method, block=adaptive_block,
                      C=adaptive_c, min_size=min_object_size,
                      smooth_sigma=mask_smooth_sigma, open_kernel=open_kernel)

    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("1. Grayscale (raw)")
        axes[1].imshow(corrected, cmap="gray"); axes[1].set_title("2. Illumination-corrected")
        axes[2].imshow(mask, cmap="gray"); axes[2].set_title("3. Foreground mask (bacteria=1)")
        for a in axes:
            a.set_xticks([]); a.set_yticks([])
        plt.tight_layout()
        figures["preprocessing"] = fig
        plt.show()

    # --- Step 2: structure tensor + windowed Q-tensor ---
    Jxx, Jyy, Jxy = structure_tensor(corrected, sigma=struct_sigma)
    field = coarse_grain_qtensor(Jxx, Jyy, Jxy, mask,
                                  window_size=window_size, overlap=overlap,
                                  min_frac=min_foreground_fraction)

    n_rows, n_cols = field["cx"].shape
    step_y, step_x = field["step_y"], field["step_x"]
    valid_grid = ~np.isnan(field["S"])
    print(f"Grid size: {n_rows} x {n_cols} windows (window={window_size}px, "
          f"overlap={overlap*100:.0f}%, step={step_y}x{step_x}px)")
    print(f"Windows with enough foreground: {valid_grid.sum()} / {valid_grid.size}")
    print(f"Mean order parameter S (over valid windows): {np.nanmean(field['S']):.3f}")

    # --- director field overlay ---
    if show_plots:
        fig, ax = plt.subplots(figsize=(9, 9))
        plot_director_field(corrected, field, rod_length_scale=rod_length_scale, ax=ax)
        plt.tight_layout()
        figures["director_field"] = fig
        plt.show()

    # --- Q-tensor / order-parameter heatmaps ---
    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plot_heatmap(field["S"], "Order parameter S (local nematic order)",
                     ax=axes[0], cmap="magma", vmin=0, vmax=1)
        vlim = np.nanmax(np.abs(np.concatenate([field["Qxx"].ravel(), field["Qxy"].ravel()])))
        plot_heatmap(field["Qxx"], "Q_xx", ax=axes[1], cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        plot_heatmap(field["Qxy"], "Q_xy", ax=axes[2], cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        plt.tight_layout()
        figures["qtensor_heatmaps"] = fig
        plt.show()

    # --- Step 3: nematic correlation function + correlation length ---
    H, W = mask.shape
    img_diag = np.sqrt(H ** 2 + W ** 2)
    r_max_cutoff = max_r_fraction_of_diagonal * img_diag

    C2D, pair_counts, center = nematic_correlation_2d(field["Qxx"], field["Qxy"], valid_grid)
    r_vals, C_vals, pairs_vals, min_pairs_used = radial_average_correlation(
        C2D, pair_counts, center,
        step_x=field["step_x"], step_y=field["step_y"],
        min_pairs_fraction=min_pairs_fraction,
        r_max=r_max_cutoff,
    )
    xi_px, r2, r_fit, C_fit, cutoff_used = fit_correlation_length(
        r_vals, C_vals, decay_threshold=decay_threshold
    )

    print(f"Correlation length xi = {xi_px:.1f} px  (fit R^2 = {r2:.3f}, "
          f"fit used r <= {cutoff_used:.0f} px, hard cutoff at {r_max_cutoff:.0f} px, "
          f"min pairs/bin = {min_pairs_used:.0f})")
    if r2 < 0.8:
        print("Warning: low R^2 -- check the log-linear panel; a curved (not "
              "straight) decay there suggests power-law/quasi-long-range order "
              "rather than a single clean xi.")

    if show_plots:
        fig, axes = plot_correlation_function(
            r_vals, C_vals, pairs_vals, r_fit, C_fit,
            xi_px, cutoff_used, r_max_cutoff, min_pairs_used
        )
        figures["correlation"] = fig
        plt.show()

    # --- optional saving ---
    if save_prefix is not None:
        save_qtensor_npz(field["Qxx"], field["Qxy"], f"{save_prefix}_Q.npz")
        np.savez(f"{save_prefix}_results.npz",
                 theta=field["theta"], S=field["S"], Qxx=field["Qxx"], Qxy=field["Qxy"],
                 cx=field["cx"], cy=field["cy"])
        print(f"Saved {save_prefix}_results.npz")

    return {
        "gray": gray, "corrected": corrected, "mask": mask,
        "field": field,
        "xi_px": xi_px, "r2": r2, "cutoff_used": cutoff_used,
        "r_max_cutoff": r_max_cutoff, "min_pairs_used": min_pairs_used,
        "r_vals": r_vals, "C_vals": C_vals, "pairs_vals": pairs_vals,
        "figures": figures,
    }
