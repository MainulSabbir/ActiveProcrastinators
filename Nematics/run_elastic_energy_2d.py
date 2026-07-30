"""
Run elastic_energy_2d.py (Emersic et al., PRL 135, 048301 (2025), Eq. 2) on the
BARCODE Q-tensor output of ANY processed dataset.

    python run_elastic_energy_2d.py <processed_dir> [--k11 0.4 --k33 1.0 --s0 1.0]
                                    [--frame N] [--save-maps]

<processed_dir> is any folder containing a Q_data/ directory of *_Q.npz files
(e.g. BARCODE_team/17-34-45_processed, BARCODE_team/processed_15-06-40, ...).
Frame numbering may start anywhere; the runner sorts by the number in the name.

Outputs into <processed_dir>/elastic_energy_2d/:
    elastic_energy_2d.csv       per-frame E_total / E_L1 / E_L2 / E_L6 summaries
    maps/<frame>_E2.npz         per-frame energy maps (only with --save-maps)

CONSTANTS ARE PHYSICS INPUT. --k11/--k33 (splay/bend Frank constants) and --s0
(reference scalar order parameter) default to the paper's demo values and are
almost certainly NOT right for a given system. Set them.
"""

import argparse
import csv
import os

import numpy as np

from barcode_energy import (energy_maps, frame_index, list_q_files, load_Q,
                            resolve_frame)
import elastic_energy_2d as ee


def summarise(m, pixel_size):
    """Scalar reductions over the non-core region of one frame's maps."""
    valid = ~m["core"]
    da = pixel_size ** 2
    Etot, E1, E2, E6 = m["E_total"], m["E_L1"], m["E_L2"], m["E_L6"]
    tot_sum = Etot[valid].sum()
    return {
        "E_total_mean": float(Etot[valid].mean()),
        "E_L1_mean": float(E1[valid].mean()),
        "E_L2_mean": float(E2[valid].mean()),
        "E_L6_mean": float(E6[valid].mean()),
        "E_total_integral": float(tot_sum * da),
        "E_L2_integral": float(E2[valid].sum() * da),
        "E_L6_integral": float(E6[valid].sum() * da),
        "L6_over_total_share": float(E6[valid].sum() / tot_sum) if tot_sum != 0 else np.nan,
        "core_fraction": float(m["core"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--k11", type=float, default=0.4, help="splay Frank constant (PLACEHOLDER)")
    ap.add_argument("--k33", type=float, default=1.0, help="bend Frank constant (PLACEHOLDER)")
    ap.add_argument("--s0", type=float, default=1.0, help="reference order parameter for K->L")
    ap.add_argument("--pixel-size", type=float, default=1.0)
    ap.add_argument("--core-threshold", type=float, default=0.15)
    ap.add_argument("--frame", type=int, default=None, help="run one frame only")
    ap.add_argument("--save-maps", action="store_true")
    args = ap.parse_args()

    L1, L2, L6 = ee.L_from_K(args.k11, args.k33, args.s0, trace=2)
    K_check = ee.K_from_L(L1, L2, L6, args.s0, trace=2)
    print(f"{args.processed_dir}")
    print(f"K11={args.k11} K33={args.k33} S0={args.s0} -> "
          f"L1={L1:.4f} L2={L2:.4f} L6={L6:.4f}  (reads back to K={tuple(round(k,4) for k in K_check)})")

    out_dir = os.path.join(args.processed_dir, "elastic_energy_2d")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_maps:
        os.makedirs(os.path.join(out_dir, "maps"), exist_ok=True)

    # which frames to process: one (--frame) or the whole dataset
    if args.frame is not None:
        files = [resolve_frame(args.processed_dir, args.frame)]
    else:
        files = list_q_files(args.processed_dir)

    # loop: load each Q file -> energy maps -> one summary row (+ optional map file)
    rows = []
    for i, path in enumerate(files):
        m = energy_maps(*load_Q(path), args.k11, args.k33, args.s0,
                        args.core_threshold, args.pixel_size)
        row = {"frame_num": frame_index(path), "filename": os.path.basename(path)}
        row.update(summarise(m, args.pixel_size))          # scalar reductions -> CSV row
        rows.append(row)

        if args.save_maps:
            np.savez_compressed(
                os.path.join(out_dir, "maps",
                             os.path.basename(path).replace("_Q.npz", "_E2.npz")),
                E_total=m["E_total"].astype(np.float32), E_L1=m["E_L1"].astype(np.float32),
                E_L2=m["E_L2"].astype(np.float32), E_L6=m["E_L6"].astype(np.float32),
                core=m["core"])
        if (i + 1) % 25 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)} frames", flush=True)

    rows.sort(key=lambda r: r["frame_num"])
    csv_path = os.path.join(out_dir, "elastic_energy_2d.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
