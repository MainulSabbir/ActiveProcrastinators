# Elastic-energy analysis of BARCODE active-nematic data

Compute and visualize the **elastic energy** of a 2D active nematic from BARCODE
Q-tensor output, following Eq. 2 and Fig. 3 of
*Sokolov, Katuri, de Pablo & Snezhko, "Synthetic Active Liquid Crystals Powered
by Acoustic Waves", Adv. Mater. 2025, 37, 2418846.*

The pipeline goes: **`Q_data/*.npz`  →  elastic energy  →  three plots.**

> **Scope.** This documents the **standalone command-line tools** in this folder,
> which work directly on an existing folder of Q-tensor `.npz` files. If you want
> to go from *raw images* instead, use the app's **Nematics Analysis** tab
> (Process Data → Nematics), which extracts the Q tensor for you and reports the
> director field, correlation length, and energy-spectrum length scale — see
> `NEMATICS_INTEGRATION.md` in the repository root.

---

## 1. What each file is

| File | Purpose | Run it? |
|------|---------|---------|
| `elastic_energy_2d.py` | The physics: Landau–de Gennes energy, Eq. 2 (`build_Q`, `elastic_energy_2d`, `L_from_K`). | No — imported |
| `barcode_energy.py` | Shared helpers: find/load `.npz`, the Q→energy **adapter**, and the spectrum math. | No — imported |
| `plot_energy_map.py` | **Plot 1** — spatial elastic-energy map (`pcolor`) for one frame. | Yes |
| `plot_energy_hist.py` | **Plot 2** — elastic-energy distribution (histogram) for one frame. | Yes |
| `plot_energy_spectrum.py` | **Plot 3** — time-averaged energy spectrum E(k) + its peak. | Yes |
| `run_elastic_energy_2d.py` | Batch: every frame → one CSV of energy summaries (+ optional maps). | Yes |

`barcode_energy.py` is the single place holding the file conventions and the
physics; the four runnable scripts all import from it, so they never disagree.

---

## 2. The data it reads

Any **processed folder** with this layout (BARCODE's output):

```
<processed_dir>/
└── Q_data/
    ├── pic--1_Q.npz
    ├── pic--2_Q.npz      each .npz holds arrays  Qxx, Qxy
    └── ...
```

BARCODE stores `Qxx = S·cos2θ`, `Qxy = S·sin2θ`, where
`S = sqrt(Qxx² + Qxy²)` is the order parameter and `θ` the director angle.
Frame numbering can start anywhere (`pic--1…` or `pic--1001…`) and images can be
any size — the scripts adapt.

Point the scripts at any such folder produced by the Nematics branch
(`Nematics_Output/`) or by BARCODE's Q-tensor export.

---

## 3. The one setup detail: which Python

numpy/scipy/matplotlib live **only** in the repo's venv. Use it
explicitly (a bare `python`/`python3` will fail with `ModuleNotFoundError`):

```bash
cd /path/to/barcode-main
V=venv/bin/python        # <- the interpreter to use for everything below
```

In VS Code: **Cmd+Shift+P → Python: Select Interpreter →** the one ending in
`venv/bin/python`, then Run/F5 works.

---

## 4. The workflow, step by step

### Step 0 — see which frames a dataset has
```bash
$V plot_energy_map.py <processed_dir> --list
# -> <processed_dir>: frames 1..227 (227 total)
```

### Plot 1 — spatial energy map (pcolor), one frame
```bash
$V plot_energy_map.py <processed_dir> --frame 89 --save map_f89.png
```
The elastic energy density `(∇Q)²` at every pixel; defect cores are blanked
white. Add `--imshow` for a fast (pixel-index) render instead of `pcolor`.

### Plot 2 — energy distribution (histogram), one frame
```bash
$V plot_energy_hist.py <processed_dir> --frame 89 --save hist_f89.png
```
The histogram of the same per-pixel energy over non-core pixels (log-y, since the
distribution is heavy-tailed), with mean and median marked.

### Plot 3 — time-averaged energy spectrum + peak, whole dataset
```bash
# steady turbulent window is more meaningful than the full transient:
$V plot_energy_spectrum.py <processed_dir> --fmin 40 --fmax 100 --save spec.png
```
Averages E(k) over the chosen frames and marks the peak wavenumber `k*`
(cyclic `k = 1/λ` in µm⁻¹, matching the article; dominant length scale
`λ* = 1/k*`). Omit `--fmin/--fmax` for all frames.

### Batch — every frame to a CSV (for time-series analysis)
```bash
$V run_elastic_energy_2d.py <processed_dir> --save-maps
# writes <processed_dir>/elastic_energy_2d/elastic_energy_2d.csv  (+ maps/*.npz)
```

Output PNGs go wherever `--save` points; a `<processed_dir>/plots/` folder is a
tidy convention.

---

## 4b. Function reference

Everything below is importable. Add this folder to your path first:

```python
import sys; sys.path.insert(0, "Nematics")
```

### `elastic_energy_2d.py` — the physics
Landau–de Gennes elastic energy, Eq. 2 of the paper above. Knows nothing about
files or BARCODE; it takes arrays and returns arrays.

| Function | What it does |
|---|---|
| `build_Q(theta, S, trace=2)` | Builds the 2×2 Q-tensor from a director angle field and order parameter: `Q_ij = S(n_i n_j − δ_ij/trace)`. Returns a nested list `Q[i][j]`, each entry a 2D array. Use `trace=2` for the 2D traceless convention BARCODE stores |
| `elastic_energy_2d(Q, dx, dy, L1, L2, L6)` | The energy itself. Returns `(E_total, E_L1, E_L2, E_L6)` — the total density plus its three separate contributions, all per-pixel maps |
| `L_from_K(K11, K33, S, trace=2)` | Converts Frank constants (splay `K11`, bend `K33`) into the Landau–de Gennes `L1, L2, L6`. Note `L1` always comes back 0: in 2D only `(2L1 + L2)` and `L6` are determined, so the whole isotropic part is carried by `L2` |
| `K_from_L(L1, L2, L6, S, trace=2)` | The inverse. Useful for sanity-checking `L_i` values quoted in a paper |
| `grad(f, dx, dy)` | Helper: returns `[df/dx, df/dy]` for an array laid out as `[y, x]` |

### `barcode_energy.py` — files, energy maps, and the spectrum
The bridge between stored `.npz` files and the physics above.

**Finding and loading frames**

| Function | What it does |
|---|---|
| `list_q_files(processed_dir)` | Every `*_Q.npz` in `<processed_dir>/Q_data`, sorted by frame number (not alphabetically, so `pic--9` precedes `pic--10`) |
| `frame_range(processed_dir)` | `(first, last, count)` of the available frame numbers |
| `resolve_frame(processed_dir, frame)` | Path to a given frame. Matches the *actual* numbering, so it works whether a dataset starts at 1 or 1001; `None` picks the middle frame |
| `frame_index(path)` | Pulls the integer frame number out of a `…--<N>_Q.npz` filename |
| `load_Q(path)` | Reads `(Qxx, Qxy)` as float64 from one `.npz` |

**Energy**

| Function | What it does |
|---|---|
| `energy_maps(Qxx, Qxy, k11, k33, s0, core_threshold=0.15, pixel_size=1.0)` | The adapter: recovers `θ` and `S` from the stored components, rebuilds the full tensor, evaluates Eq. 2, and flags defect cores. Returns a dict with `E_total`, `E_L1`, `E_L2`, `E_L6`, `S`, `theta`, `core`, and the `L` constants used |

**Spectrum** — call these three in order, or use the one-shot wrapper.

| Function | What it does |
|---|---|
| `spectrum_setup(shape, pixel_size, nbins=None)` | Precomputes the k-grid, Hann taper, and radial bin assignment **once** for a dataset. Doing this per frame is the slow path; reusing the setup is ~50× faster |
| `frame_energy_spectrum(Qxx, Qxy, setup)` | `E(k)` for one frame — the elastic energy summed into radial wavenumber shells |
| `peak_wavenumber(k_centers, E)` | The peak `k*` of a spectrum, refined to sub-bin precision by fitting a parabola to the three points around the maximum. Skips the DC bins |
| `timeavg_energy_spectrum(processed_dir, pixel_size, fmin=None, fmax=None, nbins=None)` | **One-shot**: does all three of the above across a whole dataset. Returns `(k_centers, E_avg, kstar, files)`. The length scale is `λ* = 1/kstar` |

### `polscope_qtensor_functions.py` — PolScope (multipolarized) images
Jones-calculus reconstruction from a raw 2×2 polarization mosaic.

| Function | What it does |
|---|---|
| `load_channels(path)` | Splits the raw mosaic into its four polarization channels `(I0, I45, I90, I135)` |
| `reconstruct(channels)` | The reconstruction. Returns `(alpha, delta)` — director angle and retardance in radians, per pixel |
| `order_parameter(channels)` | Mean scalar order parameter over the image, `⟨√(P1² + P2²)⟩` |
| `save_and_plot(channels, alpha, delta, stem, op=None)` | Builds `Qxx/Qxy`, writes `<stem>.npz`, and saves a four-panel figure. Returns `(Qxx, Qxy)` |
| `pick_image()` | Interactive helper: lists images in the current folder and prompts for one |

### `biofilm_qtensor.py` — microscopy images
Structure-tensor extraction plus the correlation-length machinery. The
correlation functions are method-agnostic — they work on *any* Q field, which is
why the multipolarized path reuses them.

**Extraction pipeline** (in order)

| Function | What it does |
|---|---|
| `load_grayscale(path, channel="auto")` | Loads an image as float grayscale in [0, 1], signal bright |
| `correct_illumination(gray, sigma)` | Divides out slowly-varying illumination estimated by a large Gaussian blur |
| `make_mask(corrected, ...)` | Foreground mask: signal = 1, background = 0 |
| `structure_tensor(gray, sigma)` | Sobel gradients → local structure tensor `J`, smoothed over `sigma` |
| `coarse_grain_qtensor(Jxx, Jyy, Jxy, mask, window_size, overlap, min_frac)` | Averages `J` over sliding windows (foreground only) → the director grid. Returns a dict with `Qxx`, `Qxy`, `S`, `theta`, window centres, and the grid step |
| `run_pipeline(image_path, ..., save_prefix=None, show_plots=True)` | **One-shot**: runs everything above, plus the correlation length and plots. Returns a dict with `field`, `xi_px`, `r2`, the `C(r)` arrays, and the figures |

**Correlation length** (also usable standalone)

| Function | What it does |
|---|---|
| `nematic_correlation_2d(Qxx, Qxy, valid)` | 2D autocorrelation of the Q field by FFT (Wiener–Khinchin). The mean Q is subtracted, so this is the *fluctuation* correlation — a global alignment won't inflate it |
| `radial_average_correlation(C_norm, pair_counts, center, step_x, step_y, ...)` | Bins the 2D map by radial distance, with guards against the noisy far tail where few pixel pairs contribute |
| `fit_correlation_length(r, C, decay_threshold=0.12, min_points=4)` | Fits `C(r) = A·exp(−r/ξ)` to the reliable near-field decay. Returns `ξ` in pixels with the fit `R²` |

**Output**

| Function | What it does |
|---|---|
| `plot_director_field(gray_bg, field, rod_length_scale=0.9, ax=None)` | Director overlay — rods coloured by local order `S` |
| `plot_correlation_function(...)` | The 3-panel `C(r)` diagnostic: the fit, a log-linear straightness check, and pairs-per-bin |
| `save_qtensor_npz(Qxx, Qxy, path)` | Writes the two independent components as float32, in the convention these tools expect |

### `saturation_tool.py` — SIPLI images

| Function | What it does |
|---|---|
| `analyze_saturation(image_path, outdir, bg_thresh=25)` | Splits into hue/saturation/brightness and returns the saturation min, max, and mean over the sample, with background excluded by the brightness threshold |
| `find_image_in_folder(folder)` | Picks the image to analyze from a folder, skipping the tool's own outputs |

### A minimal example

```python
import sys; sys.path.insert(0, "Nematics")
import barcode_energy as be

# length scale of a whole dataset, in microns
k, E, kstar, files = be.timeavg_energy_spectrum("path/to/processed_dir", pixel_size=0.45)
print(f"peak k* = {kstar:.4f} um^-1   ->   length scale = {1/kstar:.2f} um")

# energy maps for one frame
Qxx, Qxy = be.load_Q(be.resolve_frame("path/to/processed_dir", frame=89))
m = be.energy_maps(Qxx, Qxy, k11=0.4, k33=1.0, s0=1.0)
print("mean elastic energy:", m["E_total"][~m["core"]].mean())
```

---

## 5. Options shared by the scripts

| Flag | Default | Meaning |
|------|---------|---------|
| `--frame N` | middle frame | which frame (plots 1–2); matches the real frame numbers |
| `--fmin / --fmax N` | all | frame range to average/batch (plot 3, runner) |
| `--k11 / --k33` | 0.4 / 1.0 | Frank splay / bend constants |
| `--s0` | 1.0 | reference order parameter for K→L conversion |
| `--core-threshold` | 0.15 | mask pixels with S below this × median(S) |
| `--pixel-size` | 0.45 | µm per pixel (plot 3); the paper's σ=2px ≈ 0.9 µm |
| `--save PATH` | show | write a PNG instead of opening a window |
| `--list` | — | print the available frame range and exit |

---

## 6. Things to know before quoting numbers

- **Elastic constants are placeholders.** `--k11 0.4 --k33 1.0 --s0 1.0` are the
  paper's demo values, not calibrated to these datasets. They set the absolute
  energy scale (and the small anisotropic term); the *spatial patterns* and
  *trends* are robust to them. The reference bending constant for 15% DSCG is
  ≈ 30–37 pN if you calibrate.
- **Physical units assume 0.45 µm/px.** Plot 3 reports both µm and px; the **px**
  numbers need no assumption. Change with `--pixel-size` if your data differs.
- **The peak scale shifts with activity** (the paper's key result). Averaging
  across a transient blurs it — restrict Plot 3 to a fixed-activity window with
  `--fmin/--fmax`. Example on `17-34-45_processed`: full run λ*≈11 µm vs
  a turbulent window (frames 40–100) λ*≈7.7 µm.
- **`E_L1` is always 0** in the CSV — `L_from_K` puts the isotropic part in L2 by
  construction; that is expected, not a bug.
- **The paper's Fig. 3a energy is exactly `(∇Q)²`** — the one-constant form. The
  batch CSV's `E_total` is the full Eq. 2; the two match in the one-constant
  limit.
