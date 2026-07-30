"""
PolScope raw mosaic  ->  Q-tensor  ->  .npz
Combines Steps 1-3 (split channels, Jones reconstruction, save Qxx/Qxy).

RUN:
    python3 polscope_to_qtensor.py                 # default image below
    python3 polscope_to_qtensor.py other_image.tif # any raw mosaic

OUTPUTS (named after the input):
    <name>.npz        arrays Qxx, Qxy  (float32)
    <name>_qmaps.png  visual check (Qxx, Qxy, retardance, director)

METHOD (Jones calculus; verified against a ground-truth reconstruction):
    channel order (map C):  I0=c01, I45=c11, I90=c10, I135=c00
    A = I0 - I90 ,  B = I45 - I135
    retardance  delta = arcsin( sqrt(A^2+B^2) / (total/2) )   # radians, <= pi/2
    director    alpha = 0.5 * atan2(B, A)                     # matches reference
    Qxx = (delta/2) cos(2 alpha) ,  Qxy = (delta/2) sin(2 alpha)

NOTE: delta is a retardance in RADIANS (0..pi/2). Values > 1 are normal.
A single wavelength cannot resolve retardance above pi/2 (sin folding).
"""
import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "pic--7_cropped.tif"

# 1) load raw mosaic + split into 4 polarization channels
im = Image.open(PATH)
arr = np.asarray(im.convert("L")).astype(np.float64)
H, W = arr.shape
arr = arr[:H - H % 2, :W - W % 2]
I0, I45, I90, I135 = arr[0::2, 1::2], arr[1::2, 1::2], arr[1::2, 0::2], arr[0::2, 0::2]

# 2) Jones reconstruction -> retardance (delta) and director (alpha)
A, B = I0 - I90, I45 - I135
tot = (I0 + I45 + I90 + I135) / 2.0 + 1e-12
P1, P2 = A / tot, B / tot                       # normalized components (P~1, P~2)

# order parameter per pixel = sqrt(P1^2 + P2^2)  (equals |sin(delta)|)
order_param = np.sqrt(P1 ** 2 + P2 ** 2)
order_param_avg = order_param.mean()            # single-number summary

delta = np.arcsin(np.clip(order_param, 0, 1))
alpha = 0.5 * np.arctan2(P2, P1)

# 3) Q-tensor components + save
Qxx = (delta / 2) * np.cos(2 * alpha)
Qxy = (delta / 2) * np.sin(2 * alpha)
stem = os.path.splitext(os.path.basename(PATH))[0]
np.savez(stem + ".npz", Qxx=Qxx.astype(np.float32), Qxy=Qxy.astype(np.float32))

# report + visualize
C, S = np.mean(np.cos(2 * alpha)), np.mean(np.sin(2 * alpha))
print(f"director mean = {np.rad2deg(0.5*np.arctan2(S,C)):.1f} deg, R = {np.hypot(C,S):.3f}")
print(f"retardance mean = {delta.mean():.3f} rad, max = {delta.max():.3f}")
print(f"ORDER PARAMETER  sqrt(P1^2+P2^2)  averaged over all pixels = {order_param_avg:.4f}")
print(f"saved {stem}.npz  (Qxx, Qxy; float32; shape {Qxx.shape})")

fig, ax = plt.subplots(2, 2, figsize=(15, 9))
# Qxx, Qxy: diverging red/blue, auto-scaled
for a, img, t in [(ax[0,0], Qxx, "Qxx"), (ax[0,1], Qxy, "Qxy")]:
    im_ = a.imshow(img, cmap="RdBu_r"); a.set_title(t); a.axis("off")
    plt.colorbar(im_, ax=a, fraction=0.03)
# retardance: FIXED absolute scale so red = 1 (blue = 0).
# This makes the color meaning consistent across different images.
im_ = ax[1,0].imshow(delta, cmap="jet", vmin=0, vmax=1)
ax[1,0].set_title("retardance (rad), red = 1"); ax[1,0].axis("off")
plt.colorbar(im_, ax=ax[1,0], fraction=0.03)
ax[1,1].imshow((I0+I45+I90+I135)/4, cmap="gray")
STEP = 4   # spacing between director segments (smaller = higher resolution)
ys, xs = np.mgrid[STEP//2:alpha.shape[0]:STEP, STEP//2:alpha.shape[1]:STEP]
ph = alpha[ys, xs]
ax[1,1].quiver(xs, ys, np.cos(ph), np.sin(ph), color="red", scale=170,
               headwidth=1, headlength=0, pivot="mid", width=0.0007)
ax[1,1].set_title("director"); ax[1,1].axis("off")
# report the order parameter (and key metrics) as a header on the figure
dir_deg = np.rad2deg(0.5*np.arctan2(S, C))
fig.suptitle(
    r"Order parameter  $\langle\sqrt{P_1^2+P_2^2}\rangle$ = "
    + f"{order_param_avg:.4f}"
    + f"      |      director mean = {dir_deg:.1f} deg,  R = {np.hypot(C,S):.3f}"
    + f"      |      retardance mean = {delta.mean():.3f} rad",
    fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(stem + "_qmaps.png", dpi=200)
print(f"saved {stem}_qmaps.png")
