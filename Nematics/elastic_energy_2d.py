"""
Eq. (2) of Emersic, de Pablo, Snezhko & Sokolov, PRL 135, 048301 (2025),
written out index by index in 2D.

    E_el = sum_{i,j,k=1,2} [ (L1/2) (dQ_ij/dx_k)^2
                           + (L2/2) (dQ_ij/dx_j)(dQ_ik/dx_k) ]
         + (L6/2) sum_{i,j,k,l=1,2} [ Q_lk (dQ_ij/dx_l)(dQ_ij/dx_k) ]

Index 0 = x, index 1 = y. Arrays are indexed [y, x].
"""

import numpy as np


# ---------------------------------------------------------------------------

def build_Q(theta, S=1.0, trace=2):
    """Q_ij = S (n_i n_j - delta_ij / trace), returned as a 2x2 nested list.

    trace=2 gives the 2D traceless tensor (BARCODE output).
    trace=3 gives the 3D convention the paper's L_i actually assume.
    """
    n = [np.cos(theta), np.sin(theta)]
    d = [[1.0, 0.0], [0.0, 1.0]]
    return [[S * (n[i] * n[j] - d[i][j] / trace) for j in range(2)]
            for i in range(2)]


def grad(f, dx, dy):
    """[df/dx, df/dy] for an array laid out as [y, x]."""
    gy, gx = np.gradient(f, dy, dx, edge_order=2)
    return [gx, gy]


# ---------------------------------------------------------------------------

def elastic_energy_2d(Q, dx, dy, L1, L2, L6):
    """Energy density from Eq. (2). Q is Q[i][j], each a 2D array.

    Returns (E_total, E_L1, E_L2, E_L6).
    """
    # dQ[i][j][k] = dQ_ij / dx_k
    dQ = [[grad(Q[i][j], dx, dy) for j in range(2)] for i in range(2)]

    shape = Q[0][0].shape
    T1 = np.zeros(shape)
    T2 = np.zeros(shape)
    T3 = np.zeros(shape)

    # first term:  sum_{i,j,k} (dQ_ij/dx_k)^2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T1 += dQ[i][j][k] ** 2

    # second term:  sum_{i,j,k} (dQ_ij/dx_j)(dQ_ik/dx_k)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2 += dQ[i][j][j] * dQ[i][k][k]

    # third term:  sum_{i,j,k,l} Q_lk (dQ_ij/dx_l)(dQ_ij/dx_k)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    T3 += Q[l][k] * dQ[i][j][l] * dQ[i][j][k]

    E1 = 0.5 * L1 * T1
    E2 = 0.5 * L2 * T2
    E6 = 0.5 * L6 * T3
    return E1 + E2 + E6, E1, E2, E6


# ---------------------------------------------------------------------------
# elastic constants
# ---------------------------------------------------------------------------

def L_from_K(K11, K33, S, trace=2):
    """Frank -> L_i. Only (2 L1 + L2) and L6 are fixed in 2D, so L1 is set
    to zero and the whole isotropic part is carried by L2."""
    if trace == 3:
        L2 = (2.0 * K11 + K33) / (3.0 * S**2)
    elif trace == 2:
        L2 = (K11 + K33) / (2.0 * S**2)
    else:
        raise ValueError("trace must be 2 or 3")
    return 0.0, L2, (K33 - K11) / (2.0 * S**3)


def K_from_L(L1, L2, L6, S, trace=2):
    """L_i -> Frank. Use this to check any L_i you take from a paper."""
    iso = (2.0 * L1 + L2) * S**2
    if trace == 3:
        return iso - (2.0 / 3.0) * L6 * S**3, iso + (4.0 / 3.0) * L6 * S**3
    if trace == 2:
        return iso - L6 * S**3, iso + L6 * S**3
    raise ValueError("trace must be 2 or 3")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    K11, K33, S = 0.4, 1.0, 0.5
    N, half = 401, 6.0
    g = np.linspace(-half, half, N)
    h = g[1] - g[0]
    X, Y = np.meshgrid(g, g)
    R, PHI = np.hypot(X, Y), np.arctan2(Y, X)
    ann = (R > 1.5) & (R < 4.0)

    tests = {
        "pure splay  n = r_hat": (PHI, 0.5 * K11 / R**2),
        "pure bend   n = phi_hat": (PHI + 0.5 * np.pi, 0.5 * K33 / R**2),
        "+1/2 defect  theta = phi/2": (
            0.5 * PHI,
            (K11 * np.cos(0.5 * PHI) ** 2 + K33 * np.sin(0.5 * PHI) ** 2)
            / (8.0 * R**2)),
        "-1/2 defect  theta = -phi/2": (
            -0.5 * PHI,
            (K11 * np.cos(1.5 * PHI) ** 2 + K33 * np.sin(1.5 * PHI) ** 2)
            / (8.0 * R**2)),
    }

    for trace in (2, 3):
        L1, L2, L6 = L_from_K(K11, K33, S, trace=trace)
        print(f"\ntrace={trace}:  L1={L1:.4f} L2={L2:.4f} L6={L6:.4f}"
              f"   -> K = {tuple(round(v, 6) for v in K_from_L(L1, L2, L6, S, trace))}")
        for name, (theta, exact) in tests.items():
            Q = build_Q(theta, S=S, trace=trace)
            E = elastic_energy_2d(Q, h, h, L1, L2, L6)[0]
            rel = np.abs(E - exact)[ann] / exact[ann].mean()
            print(f"    {name:<28s} max rel err {rel.max():.2e}")

    Q = build_Q(np.full_like(X, 0.7), S=S, trace=2)
    L1, L2, L6 = L_from_K(K11, K33, S, trace=2)
    print(f"\nuniform director: max |E| = "
          f"{np.abs(elastic_energy_2d(Q, h, h, L1, L2, L6)[0]).max():.2e}")

    # what the paper's printed L_i correspond to
    Lp = ((K33 - K11) / (6 * S**2), K11 / S**2, (K33 - K11) / (2 * S**3))
    print(f"\npaper's L_i = {tuple(round(v, 4) for v in Lp)}")
    for t in (3, 2):
        print(f"   read back under trace={t}: "
              f"{tuple(round(v, 4) for v in K_from_L(*Lp, S, trace=t))}")
