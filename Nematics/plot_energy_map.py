"""
PLOT 1 of 3 -- spatial distribution of elastic energy for a single frame (pcolor).

    python plot_energy_map.py <processed_dir> --frame 89 [--save out.png]
    python plot_energy_map.py <processed_dir> --list

Elastic energy density (grad Q)^2, Eq. 2 of Sokolov et al. Adv. Mater. 2025,
computed from the BARCODE Q field. Defect cores (low S) are blanked white.
See barcode_energy.py for the physics; plot_energy_hist.py and
plot_energy_spectrum.py are the other two views.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from barcode_energy import energy_maps, frame_index, frame_range, load_Q, resolve_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--frame", type=int, default=None, help="frame number (default: middle frame)")
    ap.add_argument("--k11", type=float, default=0.4)
    ap.add_argument("--k33", type=float, default=1.0)
    ap.add_argument("--s0", type=float, default=1.0)
    ap.add_argument("--core-threshold", type=float, default=0.15)
    ap.add_argument("--imshow", action="store_true", help="use imshow instead of pcolormesh (faster)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--list", action="store_true", help="print available frame range and exit")
    args = ap.parse_args()

    # --list just reports which frames the dataset actually contains, then quits
    if args.list:
        lo, hi, n = frame_range(args.processed_dir)
        print(f"{args.processed_dir}: frames {lo}..{hi} ({n} total)")
        return

    # STEP 1 -- find the Q file for this frame and load Qxx, Qxy
    path = resolve_frame(args.processed_dir, args.frame)    # matches actual frame numbers
    frame = frame_index(path)

    # STEP 2 -- turn (Qxx, Qxy) into the elastic energy map (see barcode_energy.py)
    m = energy_maps(*load_Q(path), args.k11, args.k33, args.s0, args.core_threshold)
    E = np.where(m["core"], np.nan, m["E_total"])           # blank cores -> NaN plots blank
    print(f"{args.processed_dir}  frame {frame}:  "
          f"E mean {np.nanmean(E):.5f}  core {100*m['core'].mean():.2f}%")

    # STEP 3 -- draw the spatial map
    ny, nx = E.shape
    vmax = np.nanpercentile(E, 99)                          # clip hot tail for readability
    fig, ax = plt.subplots(figsize=(8, 6))
    if args.imshow:                                         # --imshow: fast, pixel indices
        im = ax.imshow(E, origin="upper", cmap="inferno", vmin=0, vmax=vmax)
    else:                                                   # default: pcolor (MATLAB-style)
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        im = ax.pcolormesh(X, Y, E, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.invert_yaxis()                                  # row 0 at top, image convention
    ax.set_aspect("equal")
    ax.set_title(f"elastic energy distribution   frame {frame}")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    fig.colorbar(im, ax=ax, fraction=0.045, label="elastic energy density")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=130); print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
