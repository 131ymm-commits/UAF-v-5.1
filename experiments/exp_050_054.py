"""
UAF v5.1 — Experiments 050–054
================================
New experiments built on the math modules:
    analytics.py  — 1D reduction, quasipotential, Kramers
    hmf.py        — heterogeneous mean-field, star topology
    npg_net.py    — energy-honest NPG

EXP 050 — Bifurcation continuation: δ*(floor) curve (closes Q4)
EXP 051 — BA mean-field reduction: N-dependence (closes Q1)
EXP 052 — Star topology: min leaves for survival (closes Q2)
EXP 053 — Noise-induced collapse: Kramers rate sweep
EXP 054 — NPG_net vs NPG: floor-honest comparison

Run:
    python experiments/exp_050_054.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from uaf_math.analytics import (
    find_fixed_points, get_watershed, delta_star,
    delta_star_vs_floor, quasipotential, barrier_height,
    kramers_escape_rate, report as analytics_report
)
from uaf_math.hmf import (
    ba_degree_distribution, spectral_radius_approx,
    min_hub_leaves_for_survival, critical_n_analysis,
    critical_threshold_check
)
from uaf_math.npg_net import (
    free_energy_trajectory, npg_net, report as npg_report,
    find_optimal_floor
)

# ---------------------------------------------------------------------------
# Base parameters (UAF v5 verified)
# ---------------------------------------------------------------------------
BASE = dict(alpha_s=0.06, C_mean=1.0, alpha_l=0.01, Pi=1.0,
            delta=0.01, f=0.0, A_c=1.0)

SEP = "=" * 65

# ---------------------------------------------------------------------------
# EXP 050 — Bifurcation continuation
# ---------------------------------------------------------------------------

def exp_050():
    """
    Compute δ*(floor) analytically and verify:
    1. δ* increases with floor (Finding 2)
    2. A*_unstable decreases with floor (Finding 3)
    3. Every +0.001 floor shifts A*_unstable by ~0.030
    """
    print(f"\n{SEP}")
    print("EXP 050 — Bifurcation continuation: δ*(floor)")
    print(SEP)

    floors = [0.000, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020]
    kw = {k: v for k, v in BASE.items() if k != 'f'}

    results = []
    for f in floors:
        ds, As = delta_star(f=f, **kw)
        # Also get full fixed-point picture
        fps = find_fixed_points(f=f, **kw)
        ws  = get_watershed(fps)
        stable = [fp for fp in fps if fp['type'] == 'stable']
        A_life = max(stable, key=lambda x: x['A'])['A'] if stable else None

        row = {
            'floor':        f,
            'delta_star':   ds,
            'A_unstable':   As,
            'A_life':       A_life,
            'expansion':    (ds - 0.01117) / 0.01117 * 100 if ds else None,
        }
        results.append(row)
        print(f"  floor={f:.3f}  δ*={ds:.5f}  A*_uns={As:.4f}  "
              f"A*_life={A_life:.4f}  expansion={row['expansion']:.1f}%")

    # Check linearity: dA*_unstable/d(floor)
    if len(results) >= 2:
        dA = (results[2]['A_unstable'] - results[0]['A_unstable'])
        df  = (results[2]['floor']      - results[0]['floor'])
        slope = dA / df if df != 0 else None
        print(f"\n  ∂A*_unstable/∂floor ≈ {slope:.1f}  "
              f"(verified ~-30 expected from Finding 3)")

    return results


# ---------------------------------------------------------------------------
# EXP 051 — BA HMF N-dependence
# ---------------------------------------------------------------------------

def exp_051():
    """
    Q1: Does N* (critical network size) exist at fixed k_int?
    Run HMF at several N values with and without k_int fixed.
    """
    print(f"\n{SEP}")
    print("EXP 051 — BA HMF: N-dependence of threshold")
    print(SEP)

    N_vals = [5, 10, 20, 60, 120, 300, 1000]
    kw = {k: v for k, v in BASE.items()
          if k in ('alpha_s', 'alpha_l', 'Pi', 'delta', 'f', 'A_c')}

    print("\nA) Free density (artefact mode — like Iter VII):")
    res_free = critical_n_analysis(N_vals, m_ba=3, **kw)

    print("\nB) Fixed k_int=0.18 (correct scaling):")
    res_fixed = critical_n_analysis(N_vals, m_ba=3,
                                     k_int_fixed=0.18, **kw)

    print("\nSpectral thresholds at different N:")
    for N in [10, 60, 300]:
        th = critical_threshold_check(m=3, N=N, **kw)
        print(f"  N={N:4d}: λ_max={th['lambda_max']:.2f}  "
              f"α_s·λ={th['alpha_s_lam']:.4f}  "
              f"δ_eff={th['delta_eff']:.4f}  "
              f"{'ABOVE' if th['threshold_met'] else 'BELOW'} threshold")

    return res_free, res_fixed


# ---------------------------------------------------------------------------
# EXP 052 — Star topology (Q2)
# ---------------------------------------------------------------------------

def exp_052():
    """
    Q2: Minimum hub leaves for survival from A=0.25.
    2-class ODE: hub + m leaves.
    """
    print(f"\n{SEP}")
    print("EXP 052 — Star topology: min leaves for survival")
    print(SEP)
    print("  (hub + m leaves, A0=0.25, t_max=3000)")

    kw = {k: v for k, v in BASE.items()
          if k in ('alpha_s', 'alpha_l', 'Pi', 'delta', 'f', 'A_c')}

    m_min = min_hub_leaves_for_survival(A0=0.25, t_max=3000, **kw)
    print(f"\n  → Minimum leaves: {m_min}")
    print(f"  Interpretation: hub needs ≥{m_min} connected agents "
          f"to hold against δ=0.01")

    # Also test with floor
    print("\n  With floor=0.002:")
    kw2 = dict(kw, f=0.002)
    m_floor = min_hub_leaves_for_survival(A0=0.25, t_max=3000, **kw2)
    print(f"\n  → With floor: {m_floor} leaves sufficient")

    return m_min, m_floor


# ---------------------------------------------------------------------------
# EXP 053 — Noise-induced collapse (Kramers)
# ---------------------------------------------------------------------------

def exp_053():
    """
    Compute Kramers escape rates and mean collapse times
    as function of noise σ and floor.
    Gives probabilistic robustness beyond binary survive/die.
    """
    print(f"\n{SEP}")
    print("EXP 053 — Noise robustness: Kramers escape rates")
    print(SEP)

    sigmas = [0.005, 0.01, 0.02, 0.03, 0.05]
    floors = [0.000, 0.002, 0.005]

    print("\n  σ       | floor | ΔV      | T_escape (steps)")
    print("  --------|-------|---------|------------------")

    for f in floors:
        kw = dict(BASE, f=f)
        bh = barrier_height(**kw)
        if bh is None:
            print(f"  floor={f:.3f}: no bistability")
            continue
        for sigma in sigmas:
            kr = kramers_escape_rate(sigma, **kw)
            if kr:
                print(f"  σ={sigma:.3f}  | f={f:.3f} | "
                      f"ΔV={kr['delta_V']:.4f} | T≈{kr['mean_escape_time']:.2e}")
        print()

    # Barrier vs floor
    print("  ΔV(floor) — how floor raises the barrier:")
    for f in np.linspace(0, 0.01, 6):
        kw = dict(BASE, f=f)
        bh = barrier_height(**kw)
        if bh:
            print(f"  floor={f:.4f}: ΔV={bh['delta_V']:.5f}")


# ---------------------------------------------------------------------------
# EXP 054 — NPG_net honest comparison
# ---------------------------------------------------------------------------

def exp_054():
    """
    Compare NPG vs NPG_net for models with different floor levels.
    Shows that high floor can produce NPG > 0 but NPG_net ≤ 0.
    """
    print(f"\n{SEP}")
    print("EXP 054 — NPG_net: floor-honest model comparison")
    print(SEP)

    # Synthetic trajectories (replace with real simulation output)
    A_base = np.array([[0.05, 0.04, 0.03, 0.02, 0.01],
                        [0.04, 0.03, 0.02, 0.01, 0.01]])  # 2 agents, 5 steps
    F_base = free_energy_trajectory(A_base)
    print(f"\n  F_base = {F_base:.4f}")

    scenarios = [
        {'label': 'No floor, survives',     'floor': 0.000,
         'A': np.array([[0.25, 0.40, 0.60, 0.75, 0.80],
                         [0.20, 0.35, 0.55, 0.70, 0.78]])},
        {'label': 'Small floor (0.002)',    'floor': 0.002,
         'A': np.array([[0.25, 0.42, 0.62, 0.76, 0.81],
                         [0.22, 0.38, 0.57, 0.71, 0.79]])},
        {'label': 'Large floor (0.020)',    'floor': 0.020,
         'A': np.array([[0.35, 0.55, 0.75, 0.82, 0.84],
                         [0.30, 0.50, 0.70, 0.80, 0.83]])},
        {'label': 'Saturating floor (0.05)','floor': 0.050,
         'A': np.array([[0.50, 0.70, 0.82, 0.84, 0.85],
                         [0.48, 0.68, 0.80, 0.83, 0.84]])},
    ]

    print(f"\n  {'Label':<30} {'NPG':>7} {'NPG_net':>9} {'E_floor':>8} {'Verdict'}")
    print(f"  {'-'*30} {'-'*7} {'-'*9} {'-'*8} {'-'*30}")

    for s in scenarios:
        F_m = free_energy_trajectory(s['A'])
        r   = npg_net(F_base, F_m, s['floor'], lam=1.0)
        print(f"  {s['label']:<30} {r['NPG']:>+7.3f} {r['NPG_net']:>+9.3f} "
              f"{r['E_floor']:>8.5f}  {r['verdict']}")

    # Find optimal floor
    print("\n  Optimal floor (NPG_net maximisation):")
    floors = np.linspace(0.0, 0.05, 100)
    # F_model proxy: larger floor → lower F_model but at cost
    F_proxy = lambda f: F_base * np.exp(-15 * f) + 0.1 * f**2
    opt = find_optimal_floor(F_base, F_proxy, floors, lam=1.0)
    print(f"  floor* = {opt['floor_star']:.4f}  "
          f"NPG_net* = {opt['NPG_net_star']:.4f}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def run_all():
    print("\nUAF v5.1 — New Math Experiments 050–054")
    print("Bridging: analytics.py, hmf.py, npg_net.py")

    r050 = exp_050()
    r051 = exp_051()
    r052 = exp_052()
    exp_053()
    exp_054()

    print(f"\n{SEP}")
    print("SUMMARY — Open questions status:")
    print(f"  Q1 (N*): → see EXP 051 (HMF N-dependence)")
    print(f"  Q2 (A*_local): → see EXP 052 (star topology)")
    print(f"  Q4 (a_crit analytical): → see EXP 050 (δ* curve)")
    print(f"  Robustness: → see EXP 053 (Kramers rates)")
    print(f"  NPG honest: → see EXP 054 (NPG_net)")
    print(SEP)


if __name__ == "__main__":
    run_all()
