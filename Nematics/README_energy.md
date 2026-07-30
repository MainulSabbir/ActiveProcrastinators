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
