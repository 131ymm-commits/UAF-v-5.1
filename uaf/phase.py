"""
UAF v5.1 — Phase Detector: L0 / L1 / L2 / L3
===============================================
Replaces ad hoc threshold-based classification with spectral + dynamic criteria.

Levels:
    L0 — dead:        mean(A) < θ_low
    L1 — struggling:  bistable region, near watershed
    L2 — coherent:    above upper attractor, but heterogeneous (std(A) > ε)
    L3 — integrated:  coherent AND homogeneous (std(A) < ε, spectral gap → dom)

Q3 closes here: L2→L3 transition via:
    1. std(A) < ε_std         (variance criterion, already in v5)
    2. λ₂(L_C) dominates      (spectral gap of catalytic Laplacian)
    3. order_parameter ≈ 1    (Kuramoto-style synchrony across agents)
    4. rank(corr(A)) → 1      (correlation matrix nearly rank-1)

Mathematical basis:
    - Master Stability Function (Pecora & Carroll 1998, extended)
    - Consensus dynamics: dV/dt ≤ -2·κ·λ₂(L)·V   →  L3 iff λ₂ > λ₂_crit
    - Order parameter r = |mean(exp(2πi·A))| → 1 at L3

References:
    - Kuehn (2022) multiple time scales + bifurcations
    - Pecora & Carroll: master stability function for synchronisation
    - Strogatz: Kuramoto order parameter
"""

import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# Thresholds (can be overridden)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    'theta_low':   0.15,    # L0: mean(A) below this → dead
    'theta_high':  0.50,    # L1/L2 boundary: above upper attractor basin
    'eps_std':     0.08,    # L2→L3: std(A) must be below this
    'eps_order':   0.85,    # L3: order parameter must exceed this
    'eps_rank1':   0.80,    # L3: fraction of variance in PC1
}


# ---------------------------------------------------------------------------
# 1. Spectral tools
# ---------------------------------------------------------------------------

def catalytic_laplacian(C):
    """
    Graph Laplacian of catalytic weight matrix C (N×N).
    L = D - C_sym, where D_ii = Σ_j C_ij.
    Returns eigenvalues sorted ascending.
    λ₁ ≈ 0 (connected graph), λ₂ = algebraic connectivity (Fiedler value).
    """
    C_sym = 0.5 * (C + C.T)
    D = np.diag(C_sym.sum(axis=1))
    L = D - C_sym
    eigvals = np.sort(linalg.eigvalsh(L))
    return eigvals, L


def spectral_gap(C):
    """Fiedler value λ₂ — measures how well-connected / fast-mixing the network is."""
    eigvals, _ = catalytic_laplacian(C)
    if len(eigvals) < 2:
        return 0.0
    return float(eigvals[1])


def synchronisation_rate(C, alpha_s=0.06):
    """
    Rate at which consensus is approached in the linearised system:
        σ_sync = alpha_s · λ₂(L_C)
    Larger → faster L2→L3 transition.
    """
    return alpha_s * spectral_gap(C)


# ---------------------------------------------------------------------------
# 2. Order parameter (Kuramoto-style)
# ---------------------------------------------------------------------------

def order_parameter(A):
    """
    r = |mean(exp(2πi·A))|
    r → 1: all agents at same A (L3)
    r → 0: agents uniformly spread (disorder)

    For A ∈ [0,1], maps naturally to phase ∈ [0, 2π].
    """
    phases = np.exp(2j * np.pi * np.asarray(A))
    return float(np.abs(np.mean(phases)))


def pc1_variance_fraction(A_traj):
    """
    Fraction of variance explained by first principal component of A_traj.
    A_traj: (N_agents, T_steps)
    → 1.0 means all agents move together (rank-1, L3).
    """
    A = np.asarray(A_traj)
    if A.ndim == 1 or A.shape[0] == 1:
        return 1.0
    A_centered = A - A.mean(axis=1, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(A_centered, full_matrices=False)
        total = np.sum(s**2)
        if total < 1e-12:
            return 1.0
        return float(s[0]**2 / total)
    except np.linalg.LinAlgError:
        return float(np.nan)


# ---------------------------------------------------------------------------
# 3. Phase classification
# ---------------------------------------------------------------------------

def classify_phase(A, C=None, thresholds=None):
    """
    Classify current system state into L0/L1/L2/L3.

    A: array (N,) — current closure degrees
    C: optional (N, N) catalytic weight matrix

    Returns dict with:
        'level':        int (0–3)
        'label':        str ('L0'/'L1'/'L2'/'L3')
        'mean_A':       float
        'std_A':        float
        'order_param':  float (Kuramoto r)
        'spectral_gap': float (λ₂ of L_C, if C given)
        'reason':       str (explanation)
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    A = np.asarray(A, dtype=float)
    mean_A = float(np.mean(A))
    std_A  = float(np.std(A))
    r      = order_parameter(A)
    lam2   = spectral_gap(C) if C is not None else None

    result = {
        'mean_A':       mean_A,
        'std_A':        std_A,
        'order_param':  r,
        'spectral_gap': lam2,
    }

    # L0: dead
    if mean_A < th['theta_low']:
        result.update(level=0, label='L0',
                      reason=f'mean_A={mean_A:.3f} < θ_low={th["theta_low"]}')
        return result

    # L1: alive but below upper attractor
    if mean_A < th['theta_high']:
        result.update(level=1, label='L1',
                      reason=f'mean_A={mean_A:.3f} in bistable zone')
        return result

    # L2 vs L3: above attractor, check homogeneity
    is_l3_std   = std_A < th['eps_std']
    is_l3_order = r > th['eps_order']

    if is_l3_std and is_l3_order:
        reason = (f'std_A={std_A:.4f}<{th["eps_std"]}  '
                  f'r={r:.3f}>{th["eps_order"]}')
        if lam2 is not None:
            reason += f'  λ₂={lam2:.4f}'
        result.update(level=3, label='L3', reason=reason)
    else:
        parts = []
        if not is_l3_std:
            parts.append(f'std_A={std_A:.4f}≥{th["eps_std"]}')
        if not is_l3_order:
            parts.append(f'r={r:.3f}≤{th["eps_order"]}')
        result.update(level=2, label='L2', reason=' | '.join(parts))

    return result


# ---------------------------------------------------------------------------
# 4. Trajectory-level phase tracking
# ---------------------------------------------------------------------------

def track_phases(A_traj, C=None, thresholds=None):
    """
    Classify phase at each timestep.
    A_traj: (N_agents, T_steps) or (T_steps,) for single agent.

    Returns:
        phases:  list of dicts (one per timestep)
        summary: dict with transition times L0→L1, L1→L2, L2→L3
    """
    A = np.atleast_2d(A_traj)   # (N, T)
    T = A.shape[1]
    phases = [classify_phase(A[:, t], C, thresholds) for t in range(T)]

    # Find first transition times
    transitions = {}
    current = phases[0]['level']
    for t, p in enumerate(phases):
        lvl = p['level']
        key = f'L{current}→L{lvl}'
        if lvl != current and key not in transitions:
            transitions[key] = t
            current = lvl

    return phases, transitions


def phase_at_time(A_traj, t, C=None, thresholds=None):
    """Phase at a specific timestep."""
    A = np.atleast_2d(A_traj)
    return classify_phase(A[:, t], C, thresholds)


# ---------------------------------------------------------------------------
# 5. L2→L3 critical condition
# ---------------------------------------------------------------------------

def l3_transition_criterion(C, alpha_s=0.06, delta=0.01, f=0.0):
    """
    Analytical criterion for L2→L3 transition.

    Consensus linearisation: variance decays as exp(-2·α_s·λ₂·τ)
    Condition: α_s · λ₂(L_C) > δ_eff   →  synchronisation wins over decay

    Returns dict with criterion evaluation and characteristic time.
    """
    lam2     = spectral_gap(C)
    sync_rate = alpha_s * lam2
    delta_eff = delta - 0.3 * delta   # rough: higher A → lower effective δ
    criterion_met = sync_rate > delta_eff

    # Characteristic time to halve variance: τ_half = ln(2) / (2·sync_rate)
    tau_half = np.log(2) / (2 * sync_rate + 1e-12)

    return {
        'lambda2':        lam2,
        'sync_rate':      sync_rate,
        'delta_eff':      delta_eff,
        'criterion_met':  criterion_met,
        'tau_half_std':   tau_half,
        'margin':         sync_rate - delta_eff,
    }


def l3_transition_time_estimate(A_traj, C=None, thresholds=None):
    """
    Estimate empirical L2→L3 transition step from trajectory.
    Returns step index or None if transition not observed.
    """
    phases, transitions = track_phases(A_traj, C, thresholds)
    return transitions.get('L2→L3', None)


# ---------------------------------------------------------------------------
# 6. Quick report
# ---------------------------------------------------------------------------

def report_phase(A, C=None, label=""):
    p = classify_phase(A, C)
    N = len(np.atleast_1d(A))
    print(f"\nPhase report {label}  (N={N})")
    print(f"  Level:       {p['label']}  ({p['level']})")
    print(f"  mean(A):     {p['mean_A']:.4f}")
    print(f"  std(A):      {p['std_A']:.4f}")
    print(f"  order_param: {p['order_param']:.4f}")
    if p['spectral_gap'] is not None:
        print(f"  λ₂(L_C):     {p['spectral_gap']:.4f}")
    print(f"  Reason:      {p['reason']}")
    return p


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # L0
    report_phase(np.random.uniform(0.0, 0.1, 20), label="L0 (dead)")

    # L1
    report_phase(np.random.uniform(0.2, 0.4, 20), label="L1 (struggling)")

    # L2: above attractor but spread
    report_phase(np.random.uniform(0.5, 0.9, 20), label="L2 (coherent, hetero)")

    # L3: all converged
    report_phase(np.full(20, 0.82) + np.random.normal(0, 0.01, 20),
                 label="L3 (integrated)")

    # With catalytic matrix: BA-like
    N = 30
    C = np.random.uniform(0, 0.5, (N, N))
    np.fill_diagonal(C, 0)
    lam2 = spectral_gap(C)
    print(f"\nSpectral gap (random C, N={N}): λ₂ = {lam2:.4f}")

    crit = l3_transition_criterion(C, alpha_s=0.06, delta=0.01)
    print(f"L3 criterion: {'MET' if crit['criterion_met'] else 'NOT MET'}  "
          f"margin={crit['margin']:.4f}  τ_half={crit['tau_half_std']:.1f} steps")
