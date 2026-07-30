import os
import sys

import tkinter as tk
from tkinter import ttk


def setup_main_window():
    """Create and configure the main application window"""
    root = tk.Tk()
    root.title(
        "BARCODE: Biomaterial Activity Readouts to Categorize, Optimize, Design, and Engineer"
    )
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Configure styling
    style = ttk.Style()
    style.configure("TNotebook", borderwidth=0, relief="flat")
    style.map("TNotebook.Tab", focuscolor=[("", "")])

    return root


def setup_scrollable_container(root):
    """Create scrollable container for the GUI"""
    container = ttk.Frame(root)
    container.grid(row=0, column=0, sticky="nsew")

    canvas = tk.Canvas(container, bd=0, highlightthickness=0, takefocus=0)
    v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    h_scroll = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

    v_scroll.pack(side="right", fill="y")
    h_scroll.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    scrollable_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_frame_config(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", on_frame_config)

    # Mouse wheel scrolling
    canvas.bind_all(
        "<MouseWheel>",
        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
    )
    canvas.bind_all(
        "<Shift-MouseWheel>",
        lambda e: canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"),
    )

    return scrollable_frame, canvas


def setup_log_window(root):
    """Create the processing log window and redirect stdout/stderr"""
    log_win = tk.Toplevel(root)
    log_win.title("Processing Log")

    log_frame = ttk.Frame(log_win)
    log_frame.pack(fill="both", expand=True)
    log_frame.rowconfigure(0, weight=1)
    log_frame.columnconfigure(0, weight=1)

    log_text = tk.Text(log_frame, state="disabled", wrap="word", font=("Segoe UI", 10))
    log_text.pack(side="left", fill="both", expand=True)

    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_scroll.pack(side="right", fill="y")
    log_text.configure(yscrollcommand=log_scroll.set)

    class TextRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, msg):
            try:
                self.widget.configure(state="normal")
                self.widget.insert("end", msg)
                self.widget.see("end")
                self.widget.configure(state="disabled")
            except:
                raise Exception("Program Terminated Early")

        def flush(self):
            pass

    sys.stdout = TextRedirector(log_text)
    sys.stderr = TextRedirector(log_text)

    return log_win


def _load_scaled_photo(path, max_w, max_h):
    """Load an image file and return a Tk PhotoImage scaled to fit
    (max_w, max_h), or None if it cannot be loaded (e.g. Pillow missing)."""
    try:
        from PIL import Image, ImageTk
    except Exception:
        return None
    try:
        img = Image.open(path)
        img.thumbnail((max_w, max_h))
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _add_image_viewer_tab(notebook, title, images):
    """Add a tab that shows the given images with a frame selector.

    ``images`` is a list of (label, png_path). If more than one, a dropdown
    plus Prev/Next buttons let the user step through them ("for each frame").
    Returns True if the tab was added (at least one image loaded).
    """
    tab = ttk.Frame(notebook)

    controls = ttk.Frame(tab)
    controls.pack(fill="x", padx=8, pady=(8, 4))

    img_label = tk.Label(tab)
    img_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    img_label._photo = None  # keep a reference so it is not garbage-collected

    labels = [lbl for lbl, _p in images]
    selected = tk.StringVar(value=labels[0] if labels else "")

    def show(idx):
        idx = max(0, min(idx, len(images) - 1))
        selected.set(labels[idx])
        photo = _load_scaled_photo(images[idx][1], 820, 560)
        if photo is None:
            img_label.configure(image="", text="(image unavailable)")
            img_label._photo = None
        else:
            img_label.configure(image=photo, text="")
            img_label._photo = photo  # hold reference

    # only build the selector when there is more than one frame
    if len(images) > 1:
        ttk.Label(controls, text="Frame:").pack(side="left")
        combo = ttk.Combobox(controls, values=labels, textvariable=selected,
                             state="readonly", width=32)
        combo.pack(side="left", padx=(4, 8))
        combo.bind("<<ComboboxSelected>>",
                   lambda e: show(labels.index(selected.get())))

        def step(delta):
            show(labels.index(selected.get()) + delta)

        ttk.Button(controls, text="◀ Prev", command=lambda: step(-1)).pack(side="left")
        ttk.Button(controls, text="Next ▶", command=lambda: step(1)).pack(side="left", padx=(4, 0))
        ttk.Label(controls, text=f"  ({len(images)} frames)").pack(side="left")

    # verify at least the first image loads before committing the tab
    first = _load_scaled_photo(images[0][1], 820, 560) if images else None
    if first is None:
        return False
    notebook.add(tab, text=title)
    show(0)
    return True


def setup_results_window(root, summary):
    """Display a Nematics results summary (and its plots) in its own window.

    ``summary`` is the dict returned by ``run_nematics_analysis`` (keys
    include title, text, output_dir, director_images, spectrum_image). Safe
    to call from a worker thread only via
    ``root.after(0, lambda: setup_results_window(root, summary))``.
    """
    if not summary:
        return None

    win = tk.Toplevel(root)
    win.title(summary.get("title", "Nematics Results"))
    win.geometry("900x760")

    header = tk.Label(win, text=summary.get("title", "Nematics Results"),
                      font=("Helvetica", 15, "bold"))
    header.pack(anchor="w", padx=12, pady=(12, 4))

    out_dir = summary.get("output_dir", "")
    if out_dir:
        path_lbl = tk.Label(win, text=f"Saved to: {out_dir}", fg="#555",
                            font=("Segoe UI", 9), wraplength=860, justify="left")
        path_lbl.pack(anchor="w", padx=12, pady=(0, 6))

    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    # --- Summary tab (always) ---
    summary_tab = ttk.Frame(notebook)
    text = tk.Text(summary_tab, wrap="word", font=("Menlo", 11))
    text.insert("1.0", summary.get("text", ""))
    text.configure(state="disabled")
    text.pack(side="left", fill="both", expand=True)
    s_scroll = ttk.Scrollbar(summary_tab, orient="vertical", command=text.yview)
    s_scroll.pack(side="right", fill="y")
    text.configure(yscrollcommand=s_scroll.set)
    notebook.add(summary_tab, text="Summary")

    # --- Director-field tab (per-frame viewer) ---
    director_images = [(lbl, p) for lbl, p in (summary.get("director_images") or [])
                       if p and os.path.exists(p)]
    if director_images:
        _add_image_viewer_tab(notebook, "Director fields", director_images)

    # --- Correlation-function fit tab (per-frame viewer) ---
    correlation_images = [(lbl, p) for lbl, p in (summary.get("correlation_images") or [])
                          if p and os.path.exists(p)]
    if correlation_images:
        _add_image_viewer_tab(notebook, "Correlation function", correlation_images)

    # --- Energy-spectrum tab (single time-averaged plot) ---
    spectrum_image = summary.get("spectrum_image")
    if spectrum_image and os.path.exists(spectrum_image):
        _add_image_viewer_tab(notebook, "Energy spectrum",
                              [("energy spectrum", spectrum_image)])

    # --- buttons ---
    btns = ttk.Frame(win)
    btns.pack(fill="x", padx=12, pady=(0, 12))

    def open_folder():
        import subprocess, sys as _sys
        if not out_dir:
            return
        try:
            if _sys.platform == "darwin":
                subprocess.Popen(["open", out_dir])
            elif _sys.platform.startswith("win"):
                subprocess.Popen(["explorer", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
        except Exception:
            pass

    if out_dir:
        ttk.Button(btns, text="Open Output Folder", command=open_folder).pack(side="left")
    ttk.Button(btns, text="Close", command=win.destroy).pack(side="right")

    # Bring the window to the front reliably (otherwise it can open BEHIND the
    # main BARCODE window / the "Processing Complete" dialog and go unnoticed).
    try:
        win.deiconify()
        win.lift()
        win.attributes("-topmost", True)
        win.focus_force()
        win.after(600, lambda: win.attributes("-topmost", False))
    except Exception:
        pass
    return win
