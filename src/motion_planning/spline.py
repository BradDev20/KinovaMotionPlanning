"""
Stage 1: 7-DoF joint-space B-spline trajectory with precomputed bases.

Requires: numpy, scipy (scipy.interpolate.BSpline)
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline

# ----------------------------
# Knot construction (clamped)
# ----------------------------
def make_uniform_clamped_knots(K: int, degree: int, T: float) -> np.ndarray:
    """
    Build a clamped, uniform knot vector on [0, T] for a spline with
    K control points and given degree.
    """
    p = degree
    # Number of knots = K + p + 1
    # Clamping: repeat 0 and T exactly p+1 times
    n_interior = K - p - 1
    if n_interior < 0:
        raise ValueError("K must satisfy K >= degree + 1")
    if n_interior == 0:
        interior = np.array([], dtype=float)
    else:
        interior = np.linspace(0.0, T, n_interior + 2)[1:-1]  # exclude endpoints
    knots = np.concatenate([
        np.zeros(p + 1),
        interior,
        np.full(p + 1, T)
    ])
    return knots


# ----------------------------------------
# Basis matrices at arbitrary time samples
# ----------------------------------------
def bspline_basis_matrices(times: np.ndarray, knots: np.ndarray, degree: int):
    """
    Compute basis matrices B, Bdot, Bddot of shape (N, K) at given times.
    Each column j is the value of basis function N_j(t) (or its derivatives).

    Implementation note:
      For simplicity & clarity, we create one “one-hot” spline per basis function
      and evaluate it. This is perfectly fine for K ~ 12–20 and done once
      per collocation set. You can optimize later if needed.
    """
    N = times.shape[0]
    K = len(knots) - degree - 1
    B = np.zeros((N, K))
    Bdot = np.zeros((N, K))
    Bddot = np.zeros((N, K))

    # Build each scalar basis spline N_j(t) and its derivatives
    for j in range(K):
        coeff = np.zeros(K)
        coeff[j] = 1.0
        spl = BSpline(knots, coeff, degree, extrapolate=False)
        spl_d1 = spl.derivative(1)
        spl_d2 = spl.derivative(2)

        B[:, j] = spl(times)
        Bdot[:, j] = spl_d1(times)
        Bddot[:, j] = spl_d2(times)

    # Replace NaNs (outside support) with zeros due to extrapolate=False
    B = np.nan_to_num(B)
    Bdot = np.nan_to_num(Bdot)
    Bddot = np.nan_to_num(Bddot)
    return B, Bdot, Bddot


import numpy as np
from scipy import sparse

def uniform_span_dt(K: int, degree: int, T: float) -> float:
    """
    For clamped, uniform knots on [0, T], the number of spans is S = K - degree.
    Each span has length h = T / S.
    """
    S = K - degree
    if S <= 0:
        raise ValueError("K must satisfy K > degree")
    return T / S  # h

def first_diff_matrix(K: int, degree: int, T: float) -> sparse.csr_matrix:
    """
    D1 maps control points (size K) to an approximation of continuous-time velocity
    bounds for a degree-p uniform B-spline:  (p/h) * (c_{i+1} - c_i), i=0..K-2.
    Shape: (K-1, K)
    """
    p = degree
    h = uniform_span_dt(K, degree, T)
    scale = p / h

    rows = np.arange(K - 1)
    cols_i = rows
    cols_ip1 = rows + 1
    data = np.r_[ -scale * np.ones(K - 1),  scale * np.ones(K - 1) ]
    D1 = sparse.csr_matrix((data,
                            (np.r_[rows, rows], np.r_[cols_i, cols_ip1])),
                           shape=(K - 1, K))
    return D1

def second_diff_matrix(K: int, degree: int, T: float) -> sparse.csr_matrix:
    """
    D2 maps control points (size K) to an approximation of continuous-time acceleration
    bounds for a degree-p uniform B-spline: (p*(p-1)/h^2) * (c_{i+2} - 2 c_{i+1} + c_i),
    i=0..K-3. Shape: (K-2, K)
    """
    p = degree
    h = uniform_span_dt(K, degree, T)
    scale = p * (p - 1) / (h * h)

    rows = np.arange(K - 2)
    cols_i   = rows
    cols_ip1 = rows + 1
    cols_ip2 = rows + 2

    data = np.r_[  scale * np.ones(K - 2),
                  -2*scale * np.ones(K - 2),
                   scale * np.ones(K - 2) ]
    D2 = sparse.csr_matrix((data,
                            (np.r_[rows, rows, rows],
                             np.r_[cols_i, cols_ip1, cols_ip2])),
                           shape=(K - 2, K))
    return D2


# ---------------------------------------------
# 7-DoF spline container with evaluation helpers
# ---------------------------------------------
@dataclass
class Spline7DoF:
    """
    Joint-space B-spline for a 7-DoF arm.
    - control_points: (K, 7) array
    - degree: typically 3 (cubic)
    - T: total duration (seconds)
    - knots: clamped uniform knot vector
    """
    control_points: np.ndarray  # shape (K, 7)
    degree: int
    T: float
    knots: np.ndarray

    @property
    def K(self):  # number of control points
        return self.control_points.shape[0]

    def evaluate(self, t: np.ndarray):
        """
        Evaluate q(t), qdot(t), qddot(t) at times t (shape (N,)).
        Returns:
          q:    (N, 7)
          qdot: (N, 7)
          qddot:(N, 7)
        """
        B, Bd, Bdd = bspline_basis_matrices(t, self.knots, self.degree)
        C = self.control_points  # (K,7)

        q = B @ C
        qdot = Bd @ C
        qddot = Bdd @ C
        return q, qdot, qddot

    def basis_matrices(self, t: np.ndarray):
        """
        Precompute and return (B, Bdot, Bddot) for reuse elsewhere.
        Shapes: each is (N, K).
        """
        return bspline_basis_matrices(t, self.knots, self.degree)


# --------------------
# Example (usage only)
# --------------------
if __name__ == "__main__":
    # Spline config
    degree = 3
    T = 2.0            # seconds
    K = 14             # control points per joint (typ. 12–16)

    # Build knots
    knots = make_uniform_clamped_knots(K, degree, T)

    # Example control points (K,7). Replace with your seeded path or init guess.
    # Here we just do a smooth ramp from q_start to q_goal for illustration.
    q_start = np.array([0.0, -0.5, 0.0, -1.0, 0.0, 1.0, 0.5])
    q_goal  = np.array([0.8, -0.3, 0.4, -0.6, 0.5, 1.2, 0.2])
    alphas = np.linspace(0.0, 1.0, K)[:, None]
    C = (1 - alphas) * q_start + alphas * q_goal  # (K,7)

    # Create spline
    spline = Spline7DoF(control_points=C, degree=degree, T=T, knots=knots)

    # Evaluate on a time grid
    t = np.linspace(0.0, T, 101)
    q, qdot, qddot = spline.evaluate(t)

    # If you need the basis matrices for optimization:
    B, Bdot, Bddot = spline.basis_matrices(t)
    # Then q = B @ C, etc., where C is (K,7).
