"""
PolScope -> Q-Tensor, organized as four independent functions.
Run the file: it lists the images in the current folder and lets you pick one
by number (no need to type a path). Or import the functions and call any one
of them on its own.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_channels(path):
    # Load a raw 2x2 polarization mosaic and split it into the four channels (map C order).
    arr = np.asarray(Image.open(path).convert("L")).astype(float)
    H, W = arr.shape
    arr = arr[:H - H % 2, :W - W % 2]
    I0, I45, I90, I135 = arr[0::2, 1::2], arr[1::2, 1::2], arr[1::2, 0::2], arr[0::2, 0::2]
    return I0, I45, I90, I135


def reconstruct(channels):
    # Jones inversion: turn the four channels into director angle (alpha) and retardance (delta).
    I0, I45, I90, I135 = channels
    tot = (I0 + I45 + I90 + I135) / 2 + 1e-12
    A, B = (I0 - I90) / tot, (I45 - I135) / tot
    delta = np.arcsin(np.clip(np.sqrt(A ** 2 + B ** 2), 0, 1))   # retardance (rad)
    alpha = 0.5 * np.arctan2(B, A)                                # director angle
    return alpha, delta


def order_parameter(channels):
    # Compute the scalar order parameter sqrt(P1^2+P2^2) and return its average over all pixels.
    I0, I45, I90, I135 = channels
    tot = (I0 + I45 + I90 + I135) / 2 + 1e-12
    A, B = (I0 - I90) / tot, (I45 - I135) / tot
    return float(np.sqrt(A ** 2 + B ** 2).mean())


def save_and_plot(channels, alpha, delta, stem, op=None):
    # Build the Q-tensor, save Qxx/Qxy to <stem>.npz, and show the four-panel figure.
    Qxx = (delta / 2) * np.cos(2 * alpha)
    Qxy = (delta / 2) * np.sin(2 * alpha)
    np.savez(stem + ".npz", Qxx=Qxx.astype(np.float32), Qxy=Qxy.astype(np.float32))

    I0, I45, I90, I135 = channels
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for a, img, t in [(ax[0, 0], Qxx, "Qxx"), (ax[0, 1], Qxy, "Qxy")]:
        plt.colorbar(a.imshow(img, cmap="RdBu_r"), ax=a, fraction=0.03); a.set_title(t)
    plt.colorbar(ax[1, 0].imshow(delta, cmap="jet", vmin=0, vmax=1), ax=ax[1, 0], fraction=0.03)
    ax[1, 0].set_title("retardance (rad)")
    ax[1, 1].imshow((I0 + I45 + I90 + I135) / 4, cmap="gray")
    ys, xs = np.mgrid[4:alpha.shape[0]:8, 4:alpha.shape[1]:8]; ph = alpha[ys, xs]
    ax[1, 1].quiver(xs, ys, np.cos(ph), np.sin(ph), color="red", scale=90,
                    headwidth=1, headlength=0, pivot="mid", width=0.0012)
    ax[1, 1].set_title("director")
    for a in ax.ravel():
        a.axis("off")
    if op is not None:
        fig.suptitle(f"order parameter = {op:.4f}", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    plt.savefig(stem + "_qmaps.png", dpi=150)
    plt.show()
    return Qxx, Qxy


def pick_image():
    # List the image files in the current folder and let the user choose one by number.
    files = sorted(glob.glob("*.tif") + glob.glob("*.tiff") +
                   glob.glob("*.png") + glob.glob("*.jpg"))
    if not files:
        print("No image files (.tif/.tiff/.png/.jpg) found in this folder.")
        print("Put your image in the same folder as this script and run again.")
        return None
    print("\nImages found in this folder:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    choice = input("\nPick a number: ").strip()
    try:
        return files[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None


if __name__ == "__main__":
    path = pick_image()
    if path:
        stem = os.path.splitext(os.path.basename(path))[0]
        channels = load_channels(path)
        alpha, delta = reconstruct(channels)
        op = order_parameter(channels)
        print(f"\norder parameter = {op:.4f}")
        save_and_plot(channels, alpha, delta, stem, op)
        print(f"saved {stem}.npz and {stem}_qmaps.png")
