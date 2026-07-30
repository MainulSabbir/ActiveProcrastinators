import tkinter as tk
from tkinter import ttk

from utils.gui import create_option_section, create_popup
from gui.config import BarcodeConfigGUI, InputConfigGUI


# user-facing labels <-> stored image_type values
IMAGE_TYPES = [
    ("Multipolarized (PolScope)", "multipolarized"),
    ("Microscopy", "microscopy"),
    ("Shear-Induced Polarized (SIPLI)", "sipli"),
]
_LABEL_TO_VALUE = {label: value for label, value in IMAGE_TYPES}
_VALUE_TO_LABEL = {value: label for label, value in IMAGE_TYPES}


def create_nematics_frame(parent, config: BarcodeConfigGUI, input_config: InputConfigGUI):
    """Create the Nematics Analysis tab (director field / energy spectrum /
    correlation length for multipolarized & microscopy images; saturation
    stats for SIPLI images)."""
    frame = ttk.Frame(parent)

    cm = config.modules
    nem = config.nematics_parameters

    header = ("TkDefaultFont", 15, "bold")
    frame.option_add("*font", "TkDefaultFont 13")
    row_idx = 0

    # --- enable the branch ---
    tk.Label(frame, text="Nematics Analysis", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame, row_idx, cm.nematics, "Enable Nematics Branch",
        "Analyze the selected file/directory (from the Execution Settings tab) as "
        "nematic director-field images. For Multipolarized and Microscopy images this "
        "overlays the director field and reports the correlation length, the energy "
        "spectrum plot with its peak, and the length scale. For SIPLI images it reports "
        "the min / max / average saturation only (no image output).",
    )
    row_idx += 2

    # --- image type ---
    tk.Label(frame, text="Image Type", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    type_label = tk.Label(frame, text="What kind of images are these?")
    type_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

    type_display = tk.StringVar(value=_VALUE_TO_LABEL.get(nem.image_type.get(),
                                                          IMAGE_TYPES[0][0]))
    type_menu = ttk.Combobox(
        frame, textvariable=type_display,
        values=[label for label, _ in IMAGE_TYPES],
        width=32, state="readonly",
    )
    type_menu.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
    create_popup(
        frame,
        "Multipolarized: PolScope raw polarization mosaic. Microscopy: confocal / "
        "brightfield director images. SIPLI: Shear-Induced Polarized Light Imaging "
        "(saturation only).",
        row_idx, type_label,
    )
    row_idx += 1

    # --- physics / scale parameters (multipolarized + microscopy) ---
    physics_header = tk.Label(frame, text="Director & Energy Parameters", font=header)
    physics_header.grid(row=row_idx, column=0, columnspan=3, sticky="w",
                        padx=(5, 5), pady=(10, 5))
    row_idx += 1

    physics_widgets = []

    def add_spin(label_text, var, frm, to, inc, tip, width=9):
        nonlocal row_idx
        lbl = tk.Label(frame, text=label_text)
        lbl.grid(row=row_idx, column=0, sticky="w", padx=5, pady=4)
        spin = ttk.Spinbox(frame, from_=frm, to=to, increment=inc,
                           textvariable=var, width=width)
        spin.grid(row=row_idx, column=1, sticky="w", padx=5, pady=4)
        create_popup(frame, tip, row_idx, lbl)
        physics_widgets.append((lbl, spin))
        row_idx += 1

    add_spin("Pixel size (µm/pixel)", nem.pixel_size, 1e-4, 1e3, 0.01,
             "Physical pixel size in microns. Sets the energy-spectrum wavenumber "
             "axis (k = 1/λ) and converts the correlation length to microns.")
    add_spin("Splay constant k11", nem.k11, 0.0, 1e3, 0.1,
             "Splay Frank elastic constant (energy weighting).")
    add_spin("Bend constant k33", nem.k33, 0.0, 1e3, 0.1,
             "Bend Frank elastic constant (energy weighting).")
    add_spin("Reference order S0", nem.s0, 0.0, 1.0, 0.05,
             "Reference scalar order parameter used by the elastic-energy model.")
    add_spin("Director smoothing σ (px)", nem.smoothing_sigma, 0.0, 50.0, 0.5,
             "Multipolarized only. Gaussian smoothing applied to the PolScope "
             "Q-tensor before the energy spectrum. The per-pixel Jones director "
             "estimate is noisy at the pixel scale, which otherwise pushes the "
             "spectrum peak to the Nyquist limit (~2 px) and makes the length "
             "scale meaningless. Default 2 px (~0.9 µm) follows Sokolov et al., "
             "Adv. Mater. 2025. Set 0 to disable.")
    add_spin("Coarse-grain window (px)", nem.window_size, 4, 256, 2,
             "Window size in pixels for coarse-graining the director field "
             "(structure-tensor grid / block averaging).")
    add_spin("Window overlap (0–1)", nem.overlap, 0.0, 0.95, 0.05,
             "Fraction of overlap between neighboring coarse-graining windows.")

    # --- SIPLI parameters ---
    sipli_header = tk.Label(frame, text="SIPLI Parameters", font=header)
    sipli_header.grid(row=row_idx, column=0, columnspan=3, sticky="w",
                      padx=(5, 5), pady=(10, 5))
    row_idx += 1

    bg_label = tk.Label(frame, text="Background threshold (0–255)")
    bg_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=4)
    bg_spin = ttk.Spinbox(frame, from_=0, to=255, increment=1,
                          textvariable=nem.bg_threshold, width=9)
    bg_spin.grid(row=row_idx, column=1, sticky="w", padx=5, pady=4)
    create_popup(frame,
                 "Brightness (V, 0–255) below which a pixel is treated as background "
                 "and excluded from the saturation statistics.",
                 row_idx, bg_label)
    sipli_widgets = [(bg_label, bg_spin)]
    row_idx += 1

    # --- keep the stored value in sync + show only the relevant parameters ---
    def refresh_visibility(*_args):
        value = _LABEL_TO_VALUE.get(type_display.get(), "multipolarized")
        nem.image_type.set(value)
        is_sipli = value == "sipli"
        for lbl, w in physics_widgets:
            state = "disabled" if is_sipli else "normal"
            w.config(state=state)
            lbl.config(fg="grey" if is_sipli else "black")
        physics_header.config(fg="grey" if is_sipli else "black")
        for lbl, w in sipli_widgets:
            w.config(state="normal" if is_sipli else "disabled")
            lbl.config(fg="black" if is_sipli else "grey")
        sipli_header.config(fg="black" if is_sipli else "grey")

    type_display.trace_add("write", refresh_visibility)
    refresh_visibility()

    return frame
