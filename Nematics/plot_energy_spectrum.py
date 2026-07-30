"""
PLOT 3 of 3 -- time-averaged elastic energy spectrum E(k) and its peak.
Reproduces Figure 3c of Sokolov et al., Adv. Mater. 2025, 37, 2418846.

    python plot_energy_spectrum.py <processed_dir> [--pixel-size 0.45]
                                   [--fmin N --fmax N] [--save out.png]

E(k) is the elastic energy per wavenumber shell (see barcode_energy.py). The
wavenumber k is the CYCLIC k = 1/lambda in um^-1, matching the article's axis;
the peak k* is the energy-containing scale, lambda* = 1/k*. Averaging over frames
gives the time-averaged spectrum. --pixel-size defaults to 0.45 um/px (paper's
sigma=2px ~ 0.9um); pixel-unit results need no such assumption. Restrict to a
steady window with --fmin/--fmax to match a fixed activity level.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from barcode_energy import frame_index, timeavg_energy_spectrum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--pixel-size", type=float, default=0.45,
                    help="um per pixel (default 0.45, from the paper's sigma=2px~0.9um)")
    ap.add_argument("--fmin", type=int, default=None, help="first frame number to include")
    ap.add_argument("--fmax", type=int, default=None, help="last frame number to include")
    ap.add_argument("--nbins", type=int, default=None, help="number of radial k-bins")
    ap.add_argument("--kmax", type=float, default=None,
                    help="cut the plot at this k (um^-1); default auto-trims the "
                         "pixel-noise uptick at the post-peak trough")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    # STEP 1+2 -- loop every frame in the range, FFT its Q field, average E(k),
    #             and locate the peak. All of that is in barcode_energy.py.
    #             k here is the cyclic wavenumber k = 1/lambda (um^-1), as in the article.
    k, E, kstar, files = timeavg_energy_spectrum(
        args.processed_dir, args.pixel_size, args.fmin, args.fmax, args.nbins)

    # convert the peak wavenumber to a length scale (lambda = 1/k), in um and px
    lam_um = 1.0 / kstar
    lam_px = lam_um / args.pixel_size
    print(f"time-averaged over {len(files)} frames "
          f"(frames {frame_index(files[0])}..{frame_index(files[-1])})")
    print(f"peak k*    = {kstar:.4f} um^-1   ({kstar*args.pixel_size:.4f} px^-1)")
    print(f"peak scale = {lam_um:.2f} um       ({lam_px:.1f} px)")

    # trim the high-k tail: at pixel scale E(k) turns UP again (aliasing/noise),
    # which is not physical. Cut at --kmax if given, else automatically at the
    # trough just after the peak (the last physical point before the uptick).
    ipk = 2 + int(np.argmax(E[2:]))                        # peak bin
    if args.kmax is not None:
        cut = int(np.searchsorted(k, args.kmax))
    else:
        cut = ipk + int(np.argmin(E[ipk:])) + 1           # +1 to keep the trough itself
    kplot, Eplot = k[:cut], E[:cut]

    # STEP 3 -- log-log plot of E(k) with the peak marked
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(kplot, Eplot, lw=1.8)
    ax.axvline(kstar, color="C3", ls="--", lw=1,
               label=f"peak  k*={kstar:.3f} um$^{{-1}}$\n$\\lambda$*={lam_um:.1f} um")
    ax.set_xlabel("wavenumber  k = 1/$\\lambda$  (um$^{-1}$)")
    ax.set_ylabel("elastic energy spectrum  E(k)")
    ax.set_title(f"time-averaged elastic energy spectrum\n{args.processed_dir}")
    ax.legend()
    # secondary top axis in wavelength lambda = 1/k
    to_lam = lambda kk: 1.0 / np.where(kk == 0, np.nan, kk)
    ax.secondary_xaxis("top", functions=(to_lam, to_lam)).set_xlabel(
        "wavelength  $\\lambda$ = 1/k  (um)")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=130); print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
