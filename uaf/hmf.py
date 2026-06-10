"""
UAF v5.1 — Heterogeneous Mean-Field (HMF) for BA networks
===========================================================
Closes Q1 (N* at fixed k_int) and Q2 (A*_local for leaf near hub).

Instead of simulating N agents, we track one ODE per degree class k.
For Barabási–Albert: P(k) ~ k^{-3}, k_min=m.

Master equation per degree class:
    dA_k/dτ = α_s · k · Θ(A) · (1−A_k)   [TSV: field from neighbours]
            + α_l · Π_k · PE_k · (1−A_k)  [FEP]
            + f · (1−A_k/A_c)              [floor]
            − δ · (1−0.3·A_k)             [decay]

where Θ(A) = (1/<k>) · Σ_k k·P(k)·A_k   ← mean-field activation field

Key results:
    - Critical condition: α_s · λ_max(C) > δ_eff
    - λ_max ≈ sqrt(k_max) for BA (Pastor-Satorras 2001)
    - Explains why N=60 survives while single agent dies

References:
    - Pastor-Satorras & Vespignani (2001) heterogeneous mean-field SIS
    - Gleeson (2021) approximate master equations
    - Dorogovtsev et al. (2008+) dynamics on scale-free networks
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings


# ---------------------------------------------------------------------------
# 1. BA degree distribution
# ---------------------------------------------------------------------------

def ba_degree_distribution(m, k_max, N=None):
    """
    BA network degree distribution P(k) ~ 2m²/k³ for k >= m.

    If N is given, k_max ~ N^{1/2} (for gamma=3, BA).

    Returns: (k_array, P_k_array) normalised.
    """
    if N is not None:
        k_max = max(k_max, int(np.sqrt(N)))

    k = np.arange(m, k_max + 1, dtype=float)
    P = 2 * m**2 / k**3
    P /= P.sum()   # normalise
    return k, P


def mean_k(k, P):
    return np.dot(k, P)

def mean_k2(k, P):
    return np.dot(k**2, P)

def spectral_radius_approx(k, P):
    """
    λ_max ≈ max(sqrt(k_max), <k²>/<k>) — spectral radius of BA adjacency.
    Gives critical threshold α_s·λ_max > δ.
    """
    mk  = mean_k(k, P)
    mk2 = mean_k2(k, P)
    return max(np.sqrt(k.max()), mk2 / mk)


# ---------------------------------------------------------------------------
# 2. HMF ODE system
# ---------------------------------------------------------------------------

def hmf_field(A_k, k, P):
    """
    Mean-field activation Θ = (1/<k>) Σ_k k·P(k)·A_k.
    This is the 'pressure' that class-k agents exert on their neighbours.
    """
    mk = mean_k(k, P)
    return np.dot(k * P, A_k) / mk


def hmf_rhs(tau, A_k, k, P,
            alpha_s=0.06, alpha_l=0.01, Pi=1.0,
            delta=0.01, f=0.0, A_c=1.0):
    """
    ODE RHS for HMF system.
    A_k: array of shape (len(k),) — closure degree per degree class.
    """
    Theta = hmf_field(A_k, k, P)
    dA = np.zeros_like(A_k)
    for i, ki in enumerate(k):
        tsv   = alpha_s * ki * Theta * (1 - A_k[i])
        fep   = alpha_l * Pi * A_k[i] * (1 - A_k[i])
        floor = f * (1 - A_k[i] / A_c)
        decay = delta * (1 - 0.3 * A_k[i])
        dA[i] = tsv + fep + floor - decay
    return dA


def run_hmf(A0_k, t_span, t_eval,
            m=3, k_max=50, N=None, **kwargs):
    """
    Integrate HMF system.

    A0_k: initial condition per degree class (array or scalar → uniform).
    Returns: solution object + (k, P).
    """
    k, P = ba_degree_distribution(m, k_max, N)
    if np.isscalar(A0_k):
        A0_k = np.full(len(k), A0_k)

    sol = solve_ivp(
        hmf_rhs,
        t_span, A0_k,
        args=(k, P),
        t_eval=t_eval,
        method='RK45',
        dense_output=True,
        **{kk: vv for kk, vv in kwargs.items()
           if kk in ('alpha_s', 'alpha_l', 'Pi', 'delta', 'f', 'A_c')},
    )
    return sol, k, P


def mean_A_from_hmf(sol, k, P):
    """Compute <A>(τ) = Σ_k P(k)·A_k(τ) from HMF solution."""
    return np.dot(P, sol.y)   # shape: (len(t_eval),)


# ---------------------------------------------------------------------------
# 3. Local dynamics: hub + star (Q2)
# ---------------------------------------------------------------------------

def star_rhs(tau, state, m_leaves,
             alpha_s=0.06, alpha_l=0.01, Pi=1.0,
             delta=0.01, f=0.0, A_c=1.0):
    """
    Exact 2-class ODE for a hub + m_leaves star.

    state = [A_hub, A_leaf]
    Hub   has degree k_hub = m_leaves → receives from ALL leaves.
    Leaf  has degree 1 → receives ONLY from hub.

    Closes Q2: A*_local for leaf near hub = leaf's fixed point in this system.
    """
    A_hub, A_leaf = state

    # Hub: connected to m_leaves leaves
    tsv_hub = alpha_s * m_leaves * A_leaf * (1 - A_hub)
    fep_hub = alpha_l * Pi * A_hub * (1 - A_hub)
    dA_hub  = tsv_hub + fep_hub + f * (1 - A_hub/A_c) - delta * (1 - 0.3*A_hub)

    # Leaf: connected only to hub
    tsv_leaf = alpha_s * 1.0 * A_hub * (1 - A_leaf)
    fep_leaf = alpha_l * Pi * A_leaf * (1 - A_leaf)
    dA_leaf  = tsv_leaf + fep_leaf + f * (1 - A_leaf/A_c) - delta * (1 - 0.3*A_leaf)

    return [dA_hub, dA_leaf]


def run_star(A0_hub, A0_leaf, m_leaves, t_span, t_eval, **kwargs):
    """
    Simulate hub + m_leaves star from initial conditions.
    Returns (t, A_hub_traj, A_leaf_traj).
    """
    sol = solve_ivp(
        star_rhs,
        t_span, [A0_hub, A0_leaf],
        args=(m_leaves,),
        t_eval=t_eval,
        method='RK45',
        kwargs=kwargs,
    )
    # Workaround: pass kwargs explicitly
    sol2 = solve_ivp(
        lambda t, y: star_rhs(t, y, m_leaves, **kwargs),
        t_span, [A0_hub, A0_leaf],
        t_eval=t_eval,
        method='RK45',
    )
    return sol2.t, sol2.y[0], sol2.y[1]


def min_hub_leaves_for_survival(A0=0.25, t_max=5000,
                                m_range=None, **kwargs):
    """
    Find minimum m_leaves such that hub+star survives from A0.
    Answers Q2 quantitatively.

    Survival criterion: final mean_A > 0.5 at t_max.
    """
    if m_range is None:
        m_range = range(1, 30)

    t_span = (0, t_max)
    t_eval = np.array([t_max])

    for m in m_range:
        _, A_hub_f, A_leaf_f = run_star(A0, A0, m,
                                         t_span, t_eval, **kwargs)
        mean_final = (A_hub_f[-1] + m * A_leaf_f[-1]) / (m + 1)
        survived = mean_final > 0.5
        print(f"  m_leaves={m:3d}: A_hub={A_hub_f[-1]:.3f}  "
              f"A_leaf={A_leaf_f[-1]:.3f}  mean={mean_final:.3f}  "
              f"{'✓ SURVIVED' if survived else '✗ died'}")
        if survived:
            return m

    return None


# ---------------------------------------------------------------------------
# 4. N* analysis (Q1)
# ---------------------------------------------------------------------------

def critical_n_analysis(N_values, m_ba=3, k_int_fixed=None,
                        t_max=3000, A0=0.25, **kwargs):
    """
    For each N in N_values, run HMF and check survival.
    If k_int_fixed is set, rescale alpha_s so that effective interactions
    per agent are constant (fixes the Iter VII artefact).

    Returns list of {'N', 'survived', 'final_mean_A', 'delta_star'}.
    """
    from .analytics import find_fixed_points  # relative import
    results = []
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 500)

    for N in N_values:
        k_max = max(m_ba + 1, int(np.sqrt(N)))
        k, P  = ba_degree_distribution(m_ba, k_max, N)
        lam   = spectral_radius_approx(k, P)

        kw = dict(kwargs)
        if k_int_fixed is not None:
            # Fix effective interaction rate: alpha_eff = alpha_s * <k>
            # Scale alpha_s so alpha_s * <k> = k_int_fixed
            mk = mean_k(k, P)
            kw['alpha_s'] = k_int_fixed / mk

        A0_k = np.full(len(k), A0)
        sol = solve_ivp(
            lambda t, y: hmf_rhs(t, y, k, P, **kw),
            t_span, A0_k, t_eval=t_eval, method='RK45',
        )
        mean_final = np.dot(P, sol.y[:, -1])
        survived   = mean_final > 0.5

        results.append({
            'N':            N,
            'k_max':        k_max,
            'lambda_max':   lam,
            'mean_A_final': mean_final,
            'survived':     survived,
            'alpha_s_eff':  kw.get('alpha_s', kwargs.get('alpha_s', 0.06)),
        })
        print(f"  N={N:5d} k_max={k_max:4d} λ_max={lam:.2f} "
              f"<A>={mean_final:.3f} {'✓' if survived else '✗'}")

    return results


# ---------------------------------------------------------------------------
# 5. Spectral threshold
# ---------------------------------------------------------------------------

def critical_threshold_check(m=3, k_max=100, N=None, **kwargs):
    """
    Check whether current parameters are above spectral threshold:
        α_s · λ_max > δ_eff

    Also computes effective δ_eff accounting for FEP and floor.
    """
    k, P   = ba_degree_distribution(m, k_max, N)
    lam    = spectral_radius_approx(k, P)
    alpha_s = kwargs.get('alpha_s', 0.06)
    delta   = kwargs.get('delta', 0.01)
    f       = kwargs.get('f', 0.0)

    # Effective decay reduced by floor
    delta_eff = delta - f * 0.3  # rough: floor shifts barrier
    threshold_met = alpha_s * lam > delta_eff

    return {
        'lambda_max':     lam,
        'alpha_s':        alpha_s,
        'alpha_s_lam':    alpha_s * lam,
        'delta_eff':      delta_eff,
        'threshold_met':  threshold_met,
        'margin':         alpha_s * lam - delta_eff,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE = dict(alpha_s=0.06, alpha_l=0.01, Pi=1.0,
                delta=0.01, f=0.0, A_c=1.0)

    print("\n=== Spectral threshold (BA, m=3, k_max=30) ===")
    th = critical_threshold_check(m=3, k_max=30, **BASE)
    for k, v in th.items():
        print(f"  {k}: {v}")

    print("\n=== Q2: min hub leaves for survival from A=0.25 ===")
    m_min = min_hub_leaves_for_survival(A0=0.25, t_max=3000, **BASE)
    print(f"\nMin leaves for hub survival: {m_min}")

    print("\n=== Q1: N-dependence (HMF) ===")
    critical_n_analysis([5, 10, 20, 60, 200], m_ba=3, **BASE)
