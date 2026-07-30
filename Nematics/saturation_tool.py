"""
saturation_tool.py
==================
Auto-finds an image in its own folder, splits it into Hue / Saturation /
Brightness, and reports the saturation min / max / average over the sample
(the black background is ignored). Saves a comparison figure.

Two functions:
    find_image_in_folder(...)   -> locate the image to use
    analyze_saturation(...)     -> do the HSB split + saturation stats + figure

Run:
    python saturation_tool.py
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # image-only backend (no popup window)
import matplotlib.pyplot as plt


# image types we will look for
IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ======================================================================
# FUNCTION 1 : find the image sitting in the same folder as this script
# ======================================================================
def find_image_in_folder(folder="."):
    """
    Look inside `folder` for image files and return the one to analyze.

    - Skips this script's own output file (..._saturation.png) so re-running
      doesn't accidentally pick up a previous result.
    - If exactly ONE image is found, it is used automatically.
    - If SEVERAL are found, they are listed and the user picks by number.
    - If NONE are found, an error is raised telling the user to add an image.

    Returns the full path to the chosen image file.
    """
    # collect candidate images, excluding our own saved output
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
        and not f.lower().endswith("_saturation.png")
    )

    if not files:
        raise FileNotFoundError(
            f"No image files found in {os.path.abspath(folder)}. "
            "Put your image next to this script."
        )

    if len(files) == 1:
        print(f"Using image: {files[0]}")
        return os.path.join(folder, files[0])

    # more than one -> let the user choose
    print("Multiple images found:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            choice = int(input(f"Pick one (1-{len(files)}): ").strip())
            if 1 <= choice <= len(files):
                return os.path.join(folder, files[choice - 1])
        except ValueError:
            pass
        print("  Invalid choice, try again.")


# ======================================================================
# FUNCTION 2 : split HSB and report saturation stats (sample only)
# ======================================================================
def analyze_saturation(image_path, outdir, bg_thresh=25):
    """
    Split the image into Hue / Saturation / Brightness and compute the
    saturation min, max, and average over the SAMPLE ONLY (pixels brighter
    than `bg_thresh`, so the black background does not drag the numbers down).

    Also saves a 4-panel comparison figure (Original / Hue / Saturation /
    Brightness) with the saturation report printed underneath.

    Parameters
        image_path : path to the input image
        outdir     : folder where the figure is saved
        bg_thresh  : brightness (V, 0-255) below which a pixel is treated as
                     background and excluded from the statistics

    Returns a dict with sat_min, sat_max, sat_avg, coverage_pct, and the
    figure path.
    """
    # --- load image (BGR; alpha channel, if any, is dropped) ---
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(outdir, exist_ok=True)

    # --- convert to HSV and split the three channels ---
    # OpenCV ranges: H 0-179, S 0-255, V 0-255
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # for display only
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # --- sample mask: keep only pixels brighter than the background ---
    mask = V > bg_thresh
    S_sample = S[mask].astype(np.float32)
    if S_sample.size == 0:
        raise ValueError("No sample pixels found. Lower bg_thresh.")

    # --- saturation statistics over the sample region ---
    stats = {
        "sat_min": float(S_sample.min()),
        "sat_max": float(S_sample.max()),
        "sat_avg": float(S_sample.mean()),
        "coverage_pct": float(100.0 * mask.mean()),
    }

    # --- make the hue channel viewable in color (easier to read) ---
    hue_vis = cv2.cvtColor(
        cv2.merge([H, np.full_like(S, 255), np.full_like(V, 255)]),
        cv2.COLOR_HSV2RGB)

    # --- 4-panel comparison figure ---
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    ax[0].imshow(rgb);     ax[0].set_title("Original", fontsize=13, fontweight="bold")
    ax[1].imshow(hue_vis); ax[1].set_title("Hue", fontsize=13, fontweight="bold")
    im2 = ax[2].imshow(S, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("Saturation", fontsize=13, fontweight="bold")
    im3 = ax[3].imshow(V, cmap="gray", vmin=0, vmax=255)
    ax[3].set_title("Brightness", fontsize=13, fontweight="bold")
    for a in ax:
        a.axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    # --- saturation report banner under the figure ---
    report = (
        f"SATURATION (sample only, background V\u2264{bg_thresh} ignored, "
        f"{stats['coverage_pct']:.1f}% of image)\n"
        f"Min {stats['sat_min']:.1f}/255  ({stats['sat_min']/255*100:.1f}%)      "
        f"Max {stats['sat_max']:.1f}/255  ({stats['sat_max']/255*100:.1f}%)      "
        f"Average {stats['sat_avg']:.1f}/255  ({stats['sat_avg']/255*100:.1f}%)"
    )
    fig.suptitle(f"HSB Comparison + Saturation  \u2014  {base}",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, report, ha="center", va="top", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f4f4", edgecolor="#888"))
    fig.tight_layout()

    # --- save ---
    path = os.path.join(outdir, f"{base}_saturation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    stats["figure"] = path
    return stats


# ======================================================================
# driver: find image next to the script, run the analysis
# ======================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = find_image_in_folder(script_dir)
    outdir = os.path.join(script_dir, "output")

    s = analyze_saturation(image_path, outdir)

    print("\n===== SATURATION (sample only, 0-255) =====")
    print(f"  min {s['sat_min']:.1f} | max {s['sat_max']:.1f} | avg {s['sat_avg']:.1f}")
    print(f"  sample coverage: {s['coverage_pct']:.1f}% of image")
    print(f"\nSaved figure: {s['figure']}")
