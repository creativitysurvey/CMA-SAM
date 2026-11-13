"""Analytical & Numerical Verification Suite for CMA-SAM (Option A)."""
from __future__ import annotations
import math
import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    h_f, w_f = 16, 16
    s_map = 3.0 * np.tanh(rng.standard_normal((h_f, w_f)))
    tau = 0.1

    logits = (s_map / tau) - np.max(s_map / tau)
    p0 = np.exp(logits) / np.sum(np.exp(logits))
    grid_m = np.arange(1, w_f + 1, dtype=np.float64)[None, :]
    x0 = float(np.sum(p0 * grid_m))
    grad_x = (1.0 / tau) * p0 * (grid_m - x0)

    eps = 1e-6
    s_p, s_n = s_map.copy(), s_map.copy()
    s_p[5, 7] += eps
    s_n[5, 7] -= eps
    xp = float(np.sum((np.exp(s_p / tau) / np.sum(np.exp(s_p / tau))) * grid_m))
    xn = float(np.sum((np.exp(s_n / tau) / np.sum(np.exp(s_n / tau))) * grid_m))
    assert abs(grad_x[5, 7] - (xp - xn) / (2 * eps)) < 1e-7

    w_tau = math.log(math.expm1(0.1 - 1e-4))
    assert abs((math.log1p(math.exp(w_tau)) + 1e-4) - 0.1) < 1e-12
    print("[PASS] Option A Spatial Softmax Jacobian & bounded activation checks passed.")


if __name__ == "__main__":
    main()
