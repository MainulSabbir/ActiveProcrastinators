"""
PLOT 2 of 3 -- distribution (histogram) of elastic energy for a single frame.

    python plot_energy_hist.py <processed_dir> --frame 89 [--save out.png]
    python plot_energy_hist.py <processed_dir> --list

Histogram of the per-pixel elastic energy density (grad Q)^2 over non-core
pixels, log-y (the distribution is heavy-tailed). Companion to plot_energy_map.py
(the same energy, shown spatially) and plot_energy_spectrum.py.
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
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--save", default=None)
    ap.add_argument("--list", action="store_true", help="print available frame range and exit")
    args = ap.parse_args()

    # --list just reports which frames the dataset actually contains, then quits
    if args.list:
        lo, hi, n = frame_range(args.processed_dir)
        print(f"{args.processed_dir}: frames {lo}..{hi} ({n} total)")
        return

    # STEP 1 -- find the Q file for this frame and load Qxx, Qxy
    path = resolve_frame(args.processed_dir, args.frame)
    frame = frame_index(path)

    # STEP 2 -- compute the energy map, then keep only non-core pixels as a 1D list
    m = energy_maps(*load_Q(path), args.k11, args.k33, args.s0, args.core_threshold)
    E = m["E_total"][~m["core"]]                            # boolean index -> flat vector
    print(f"{args.processed_dir}  frame {frame}:  "
          f"E mean {E.mean():.5f}  median {np.median(E):.5f}  core {100*m['core'].mean():.2f}%")

    # STEP 3 -- histogram those values (this IS the energy distribution)
    vmax = np.percentile(E, 99.5)                           # trim extreme tail for a readable range
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(E, bins=args.bins, range=(0, vmax), color="C3", edgecolor="none")
    ax.set_yscale("log")                                   # heavy-tailed distribution
    ax.axvline(E.mean(), color="k", ls="--", lw=1, label=f"mean {E.mean():.4f}")
    ax.axvline(np.median(E), color="0.4", ls=":", lw=1, label=f"median {np.median(E):.4f}")
    ax.set_title(f"elastic energy distribution   frame {frame}")
    ax.set_xlabel("elastic energy density")
    ax.set_ylabel("pixel count")
    ax.legend()

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=130); print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
