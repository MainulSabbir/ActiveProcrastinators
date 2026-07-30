# Nematics → BARCODE Integration — Context & Progress

> Living document. **Updated after every change.** Purpose: give full context to resume work at any time.
> Last updated: 2026-07-21 (initial scoping, pre-implementation)

---

## 1. Goal (user request, verbatim intent)

Incorporate all code in the `Nematics/` folder into the BARCODE app. Extend the
existing GUI so the user provides a **directory/folder OR a single image file**
(already supported in BARCODE) and then chooses **what kind of images** they are
analyzing. Three image types (from the "BARCODE_team" datasets):

1. **Multipolarized images** (PolScope raw 2×2 polarization mosaic)
2. **Microscopy images** (confocal / brightfield director-field images)
3. **SIPLI** — Shear-Induced Polarized Light Imaging

### Required outputs per type
| Image type | Outputs |
|---|---|
| Multipolarized | Image with **director field overlaid** + metrics: **correlation length** (from director), **length scale** (from energy spectrum), **plot of the energy spectrum with its peak** |
| Microscopy | Same as Multipolarized |
| SIPLI | **min / max / average saturation** only — **no image output** |

Final deliverable: push to **GitHub as a fork**.

---

## 2. Repository layout (as found)

Working dir: the `barcode-main` source folder.

### BARCODE app (tkinter, config-driven)
- `main.py` — entry point; builds window, `switch_page("home"|"process"|"combine")`.
- `gui/pages/home_page.py` — "BARCODE" title + two buttons: **Process Data**, **Analyze Existing Barcodes**.
- `gui/pages/processing_page.py` — Process Data page; notebook of tabs; runs `core.pipeline.run_analysis`.
- `gui/pages/analysis_page.py` — "Analyze Existing Barcodes" page.
- `gui/frames/process/execution_tab.py` — **file/dir selection lives here** (`browse_file`, `browse_folder`, radiobuttons `mode=file|dir`), channel/metadata/branch options.
- `gui/config.py` — GUI wrapper dataclasses (`InputConfigGUI`, etc.) mirroring `core/config.py`.
- `core/config.py` — plain dataclass configs (`InputConfig` has `file_path`, `dir_path`, `mode`, `configuration_file`, `length_units`, `time_units`).
- `core/pipeline.py` — `run_analysis(dir_name, config, input_config)`.
- `utils/` — reader, writer, visualization, gui helpers. `utils/gui.py` has `create_option_section`, `create_popup`.
- `requirements.txt` — imageio, matplotlib, nd2, numpy, opencv, PyYAML, scipy, scikit-image.

### Nematics/ scripts (standalone; the code to integrate)
| File | Role |
|---|---|
| `polscope_to_qtensor.py` | PolScope raw mosaic → Qxx/Qxy npz + qmaps PNG (director, retardance). Jones reconstruction. |
| `polscope_qtensor_functions.py` | Same steps as 4 importable funcs: `load_channels`, `reconstruct`, `order_parameter`, ... |
| `biofilm_qtensor.py` | **Microscopy pipeline.** `run_pipeline(image_path, ...)` → grayscale → illum-correct → mask → structure tensor → coarse-grained Q field → **director overlay** + **nematic correlation length ξ (px)**. Returns dict incl. `field{Qxx,Qxy,S,...}`, `xi_px`, figures. |
| `barcode_energy.py` | **Shared energy pipeline.** Needs `Q_data/*_Q.npz`. Key funcs: `load_Q`, `energy_maps`, `spectrum_setup`, `frame_energy_spectrum`, `peak_wavenumber`, `timeavg_energy_spectrum`. |
| `elastic_energy_2d.py` | Physics: `build_Q`, `elastic_energy_2d` (Eq. 2, PRL 135, 048301). |
| `run_elastic_energy_2d.py` | CLI driver over a processed dir's `Q_data/`. |
| `plot_energy_map.py` / `plot_energy_hist.py` / `plot_energy_spectrum.py` | CLI plotters (spectrum → E(k) + peak k*, λ*=1/k*). |
| `saturation_tool.py` | **SIPLI pipeline.** `analyze_saturation(image_path, outdir, bg_thresh=25)` → HSB split, saturation min/max/avg over sample (bg excluded). `find_image_in_folder`. |
- `ExampleNematicsData/` — sample images (`pic--*.tif`, `jitdata.tiff`, `img_000000002_Red_000.tif`).

### Data-flow mapping (how the pieces connect)
```
Multipolarized → polscope reconstruction → Qxx,Qxy ─┐
Microscopy     → biofilm structure-tensor → Qxx,Qxy ┼→ director overlay
                                                     ├→ correlation length ξ  (biofilm C(r) machinery)
                                                     └→ energy spectrum E(k) + peak (barcode_energy) → length scale λ*=1/k*
SIPLI          → saturation_tool → min/max/avg saturation (no image)
```

---

## 3. Decisions (answered by user 2026-07-21)
1. **GUI placement**: ✅ **New branch inside the Process Data page** (a new notebook tab "Nematics Analysis" + a `nematics` module toggle). Reuses existing file/dir selection in Execution Settings.
2. **Directory handling**: ✅ **One time-averaged energy spectrum** per run (folder treated as a time series), **plus a director overlay per image**. Correlation length reported per-image and averaged.
3. **Physics/scale params**: ✅ **Expose in the GUI** — pixel size (µm/px), k11, k33, s0 (and SIPLI background threshold, microscopy window/overlap).
4. **GitHub fork**: user had no preference → **confirm upstream URL + GitHub username at the end**; not blocking the build.

### Technical decisions (Claude)
- Import the existing `Nematics/` scripts unchanged by adding the folder to `sys.path` inside `analysis/nematics.py` — literally "uses all the code in the Nematics folder", original CLI `__main__` blocks preserved.
- Correlation length for **multipolarized** (PolScope): coarse-grain the per-pixel Q into a grid, then reuse biofilm's `nematic_correlation_2d` → `radial_average_correlation` → `fit_correlation_length`. **✅ User-confirmed (2026-07-21):** reusing biofilm's C(r)/fit machinery on the PolScope-derived Q field is intended. NOTE the split: the **director field itself is extracted by `polscope_qtensor_functions`** (Jones reconstruction), NOT biofilm — biofilm is used only for the correlation length + the director-overlay/correlation plots, which are method-agnostic (they only need a Q field). The microscopy path is the only one that uses biofilm for the director *extraction* (structure tensor).
- **Microscopy**: `biofilm.run_pipeline(show_plots=False)` → its `field`, `xi_px`; build director overlay via `plot_director_field` (coarse window-grid rods). Keeps the correlation-function plot.
- **Multipolarized director overlay (updated 2026-07-21)**: rendered at **full per-pixel resolution** via `_save_fine_director_overlay` — samples the per-pixel PolScope director at ~140 rods across the short axis (≈7px spacing on the examples, vs the old 16px coarse grid) and draws all rods as ONE `LineCollection` (fast even when dense), colored by the per-pixel order field. The coarse grid is now used ONLY for the ξ correlation-length metric, not for display.
- **Energy spectrum** (both types): `barcode_energy.spectrum_setup` / `frame_energy_spectrum` / `peak_wavenumber` on the Q field; time-average across a folder by resampling each frame's E(k) onto a common k-grid (`np.interp`), handling differing image sizes. Length scale λ* = 1/k* (µm). ξ converted px→µm via grid step × µm/px.
- **SIPLI**: `saturation_tool.analyze_saturation` logic but **no figure saved** — write min/max/avg saturation to CSV/log only.
- Verified 2026-07-21: all four underlying pipelines run headless on `ExampleNematicsData/` (polscope, biofilm, saturation, spectrum).

## 4. Implementation checklist
- [x] `core/config.py`: `NematicsConfig` + `nematics` flag in `ModuleConfig`; added `nematics_parameters` to `BarcodeConfig`; registered in `GUI_CONFIG_CLASSES`.
- [x] `core/__init__.py`: export `NematicsConfig`.
- [x] `gui/config.py`: `NematicsConfigGUI`; added `nematics` to `ModuleConfigGUI`; added `nematics_parameters` to `BarcodeConfigGUI`.
- [x] `analysis/nematics.py`: orchestrator `run_nematics_analysis(...)`; registered in `analysis/__init__.py`.
- [x] `gui/frames/process/nematics_tab.py`: new tab (image-type dropdown + params, show/hide by type).
- [x] `gui/pages/processing_page.py`: registered "Nematics Analysis" tab; worker routes to nematics when enabled (nematics-only runs skip the channel-required check).
- [x] `requirements.txt`: added `tifffile`, `Pillow`.
- [x] Headless tests on `ExampleNematicsData/` — all pass (see log).
- [x] Launched the real GUI on the user's display (auto-close smoke test): window opens, Process Data page builds incl. the Nematics Analysis tab, closes cleanly, no errors.
- [ ] Full interactive click-through by the user (optional confirmation).
- [ ] GitHub fork/push — deferred at user request; upstream is `github.com/BARCODE-HTP/barcode`. `gh` CLI not installed; needs user's GitHub username + auth. No git initialized yet (working folder untouched).

## 5. What the branch produces (outputs)
Output folder: `<input>/Nematics_Output/` (next to the file, or inside the folder).
Every run also writes **`nematics_summary.txt`** (the exact text shown in the GUI results
window + Processing Log) and pops up a **GUI results window** (`setup_results_window` in
`gui/window.py`) with an "Open Output Folder" button and a **tabbed plot viewer**:
- **Summary** tab — the metrics text.
- **Director fields** tab — each frame's director overlay with a **Frame dropdown + Prev/Next**
  buttons (steps through every frame). Multipolarized/microscopy only.
- **Correlation function** tab — biofilm's 3-panel C(r) fit diagnostic **per frame** (same dropdown/Prev/Next).
  **Microscopy only** — multipolarized does NOT produce this plot (user request 2026-07-21); the ξ metric is still reported.
- **Energy spectrum** tab — the time-averaged E(k) plot with its peak.
Images are the saved PNGs, loaded via Pillow (`PIL.ImageTk`) and scaled to fit; the viewer
degrades to text-only if Pillow/ImageTk is unavailable. `_add_image_viewer_tab` / `_load_scaled_photo`.
Per-image files written: `<stem>_director.png`, `<stem>_correlation.png` (+ shared `energy_spectrum.png`).
- **Multipolarized / Microscopy**: `<stem>_director.png` per image (director overlay);
  `energy_spectrum.png` + `energy_spectrum.npz` (one time-averaged E(k) with peak k*);
  `nematics_metrics.csv` (per-image **order parameter**, correlation length px & µm, fit R²,
  plus run-level **average order parameter**, mean correlation length, peak k*, length scale λ*=1/k*).
  - **Average order parameter — MULTIPOLARIZED ONLY** (user request 2026-07-21): computed via
    `polscope_qtensor_functions.order_parameter` (mean of √(P1²+P2²) over pixels). Microscopy does
    NOT report it (gated by `report_op = image_type=="multipolarized"`; microscopy frame returns
    `order_parameter=None`, and it is dropped from the summary text, metrics dict, and CSV columns).
- **SIPLI**: `sipli_saturation.csv` (per-image min/max/avg saturation + coverage). No image.

### Energy spectrum — resolution (updated)
- **Both** image types now compute E(k) on a **full-resolution (per-pixel) Q field** (spacing = pixel_size, 1 px).
  - Multipolarized: per-pixel Q straight from the PolScope Jones reconstruction.
  - Microscopy: per-pixel Q rebuilt from biofilm's structure tensor (`_per_pixel_Q_microscopy`,
    same orientation convention as `coarse_grain_qtensor`), **band-limited by a Gaussian of width
    `struct_sigma`** (the structure-tensor smoothing scale, auto-read from `biofilm_qtensor.run_pipeline`).
    Rationale: raw per-pixel Sobel director estimates are white-noise-like at the pixel scale, so
    `(∇Q)²` rises to Nyquist and pins the peak there; you can't resolve director structure finer than
    the gradient-smoothing length, so the low-pass removes only aliased noise. Verified on the example
    image: raw per-pixel → λ*≈0.9 µm (Nyquist, meaningless); coarse grid → λ*≈205 µm (too coarse);
    band-limited per-pixel → **λ*≈9.8 µm (physical)**. Do NOT re-apply the hard mask after smoothing
    (the step edge re-injects high-k energy).
- The director **overlay** and **correlation length** still use biofilm's coarse grid (window_size/overlap).

### Known limitations / tunables
- Correlation length is the **fluctuation** correlation length (mean Q subtracted), per biofilm's method.
- All frames in one run share a fixed 50-bin k-grid so E(k) can be averaged across differing image sizes.
- The spectrum peak-finder (`barcode_energy.peak_wavenumber`) is a global argmax skipping the DC bins;
  it works now that the microscopy field is band-limited. The Nematics CLI's `--kmax` only trims the plot.

## 6. How to use (in the app)
1. Home → **Process Data**.
2. **Execution Settings** tab → pick a file or a directory (as before).
3. **Nematics Analysis** tab → check **Enable Nematics Branch**, choose the **Image Type**
   (Multipolarized / Microscopy / SIPLI), set pixel size / k11 / k33 / S0 / window / overlap
   (or SIPLI background threshold). Leave the other branch checkboxes off for a Nematics-only run.
4. Click **Process Data**. Results land in `Nematics_Output/`.

## 7. Progress log
- **2026-07-21 (second machine)**: **Fixed: multipolarized length scale was pinned to Nyquist.**
  The multipolarized path fed the RAW per-pixel Jones-reconstructed Q to the energy
  spectrum, so `(∇Q)²` climbed to Nyquist and λ* pinned at 2.02 px (0.909 µm) — the exact
  pathology this doc already records as fixed for *microscopy* (`_per_pixel_Q_microscopy`
  band-limits by `struct_sigma`); multipolarized never got the equivalent. σ-sweep on
  `pic--7_cropped.tif`: σ=0 → 0.909 µm (Nyquist) | σ=1 → 23.0 | σ=2 → 24.3 | σ=3 → 27.4 —
  any smoothing moves it into a stable physical plateau.
  **Fix, per user: exposed as a GUI parameter** rather than hardcoded —
  `NematicsConfig.smoothing_sigma` (default **2.0 px ≈ 0.9 µm**, matching the Methods of
  Sokolov et al., Adv. Mater. 2025 for exactly this PolScope data; 0 disables).
  Wired through `core/config.py` → `gui/config.py` (`NematicsConfigGUI` decl/`__post_init__`/
  `config`/`update_gui`) → `gui/frames/process/nematics_tab.py` ("Director smoothing σ (px)"
  spinbox with tooltip) → `analysis/nematics.py:_process_multipolarized_frame`.
  **Scope is deliberately surgical — spectrum only.** The director overlay still uses the raw
  per-pixel field (display resolution unchanged) and ξ still uses the raw coarse grid
  (window-averaging already suppresses pixel noise). Verified: across σ=0/2/4 the ξ (93.40 µm)
  and order parameter (0.6837) are byte-identical while only λ* moves. Regression: all three
  image types OK (multipolarized λ*=24.31 µm, microscopy λ*=18.76 µm unchanged, SIPLI
  sat_avg=149.6); GUI var + config round-trip + tab build + YAML persistence all pass.
  Physical cross-check: `pic--7_cropped` is frame 7 of the same experiment as
  `17-34-45_processed` (pre-onset/quiescent); its λ*≈24 µm vs ≈7.7 µm for that run's
  turbulent window is consistent — larger structures before the instability breaks them up.
- **2026-07-21 (second machine)**: Repo transplanted to another machine; the venv did not
  survive the copy. Recreated (Python 3.12.7, `requirements.txt` + `av` per the README note)
  and re-verified the integration end-to-end there.
- **2026-07-21**: **Average order parameter is now MULTIPOLARIZED-ONLY** (user request). Gated via `report_op`; microscopy frame returns `order_parameter=None`; dropped from summary text, metrics dict, and CSV (header + avg row) for microscopy. Smoke test: gating correct (multipolarized has it in metrics/text/CSV; microscopy has none in all three); full regression over all example images = 33/33 pass, 0 crashes; GUI renders both (multipolarized summary shows the OP line, microscopy shows none). NOTE: accidentally deleted `ExampleNematicsData/Nematics_Output/` (mistook the user's own app-run outputs for test residue) — regenerable by re-running; source images untouched.
- **2026-07-21**: **Fixed "can't see the plots in the GUI while running."** Root cause: the results window (built correctly via the worker-thread → `parent.after(0, setup_results_window)` path — reproduced and confirmed it builds with image tabs) was opening **behind** the main BARCODE window / the "Processing Complete" dialog. Fix in `gui/window.py:setup_results_window`: `deiconify()` + `lift()` + `attributes("-topmost", True)` + `focus_force()`, then release topmost after 600ms. **Removed ALL titles from every plot** (user request): director overlays (`_save_fine_director_overlay`, and cleared biofilm's built-in title in `_save_director_overlay`), energy spectrum (`_save_spectrum_plot`), and the 3-panel correlation (`_save_correlation_plot` clears every axis title). Verified by spying on `Figure.savefig`: all axis titles + suptitles are empty strings across director/spectrum/correlation for both image types. GUI smoke test + window-raise both pass.
- **2026-07-21**: Explored both codebases; wrote context file; got answers to 3/4 questions (GitHub deferred).
- **2026-07-21**: Implemented full integration (config, GUI wrapper, tab, orchestrator, worker routing, requirements). Headless-verified: multipolarized single-file, microscopy 2-image folder (time-averaged spectrum + per-image overlays + metrics CSV), SIPLI color image (saturation CSV, no image), tkinter GUI construction + config round-trip, YAML save/load with new sub-config, and full app-module import. All pass.
- **2026-07-21**: Launched the real GUI on the user's display (auto-close smoke test) — opens, builds Process Data + Nematics tab, closes cleanly.
- **2026-07-21**: Added **average order parameter** (multipolarized → `polscope_qtensor_functions.order_parameter`; microscopy → mean S) to per-image + run-level metrics/CSV. Added **GUI results window** (`gui/window.py:setup_results_window`) + **`nematics_summary.txt`**: `run_nematics_analysis` now returns a summary dict; the worker passes it via `results_callback` → `parent.after(0, ...)` to render on the main thread. Re-tested all paths headless + GUI construction — all pass.
- **2026-07-21**: Switched the **microscopy energy spectrum to a full-resolution (per-pixel) Q field** (`_per_pixel_Q_microscopy`), band-limited by `struct_sigma`, replacing the coarse-grid spectrum. Peak now λ*≈9.8 µm (was 205 µm coarse / 0.9 µm raw-per-pixel). Empirically compared raw/σ=2/4/8/16/coarse before choosing. Fixed a bug where re-masking after smoothing re-pinned the peak to Nyquist. Both image types now use per-pixel spectra. Band-limit σ kept **automatic** (tracks `struct_sigma`), per user.
- **2026-07-21**: **Plots now shown in the GUI.** `run_nematics_analysis` returns `director_images` (per-frame) + `spectrum_image`; `setup_results_window` renders a tabbed viewer — Summary / Director fields (per-frame dropdown + Prev/Next) / Energy spectrum — via Pillow `ImageTk`, scaled to fit, text-only fallback. Verified: 2-frame microscopy run → 3 tabs, images load/scale, per-frame stepping; GUI smoke test + full import — all pass.
- **2026-07-21**: **Full smoke test over ALL example images** (folder grew to 7: added `CNC.tiff` (color SIPLI) + `pic--1011_cropped.tif`). Single-file matrix 7×3 = **21/21 pass, 0 crashes**. Folder mode + GUI-window build for multipolarized/microscopy/SIPLI all pass (4 tabs / Summary-only respectively). Degenerate wrong-type case (PolScope/color image → microscopy extractor) **degrades gracefully** to "No results" + Summary-only tab, no crash. `CNC.tiff` confirmed the intended color-SIPLI case (sat_avg≈150/255). NaN metrics only occur when an image is run through the wrong extractor — never on its correct type.
- **2026-07-21**: **User-confirmed** reuse of biofilm C(r)/fit machinery on the PolScope Q field (see §3). Clarified the director-extraction split (polscope vs biofilm) in this doc.
- **2026-07-21**: **Multipolarized tweaks (user request):** (1) removed the correlation-function plot for multipolarized (no `*_correlation.png`, no "Correlation function" tab; ξ metric still reported) — microscopy keeps it; (2) **higher-resolution director field** for multipolarized via new `_save_fine_director_overlay` (per-pixel director sampled ~7px vs old 16px, single fast `LineCollection`, colored by per-pixel order). Verified: multipolarized → tabs Summary/Director fields/Energy spectrum (no corr), ξ=7.02µm kept, 1.0s/2imgs; microscopy → 4 tabs incl. correlation unchanged; GUI smoke test passes.
- **2026-07-21**: **Added the correlation-function fit plot** as a per-frame tab. Refactored `_correlation_length_px` → `_correlation_data` (returns full C(r) fit data); new `_save_correlation_plot` uses biofilm's `plot_correlation_function` (3-panel). Both frame types save `<stem>_correlation.png`; summary carries `correlation_images`; results window adds a "Correlation function" tab. Verified: 2-frame multipolarized run → 4 tabs (Summary/Director fields/Correlation function/Energy spectrum), correlation PNGs generated for every frame; GUI smoke test + imports pass. Pending: full interactive click-through by user + GitHub fork.
