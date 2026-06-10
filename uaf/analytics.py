"""
UAF v5.1 — Analytical 1D Reduction
====================================
Closes Q4: a_crit = A*_unstable(params), not ad hoc 0.75.

Mean-field reduction of the master equation to a single ODE:
    dA/dτ = β·A²·(1−A) − δ·(1−0.3·A) + f·(1−A/A_c)

where β = α_s · <C> · (1 + α_l·Π/α_s) encodes both TSV and FEP contributions.

Fixed points: roots of f(A) = 0.
Bifurcation: saddle-node when f(A) = 0 AND f'(A) = 0 simultaneously.

References:
    - Strogatz (2018), §3.1 saddle-node normal form dx/dt = r + x²
    - Ashwin et al. (2012+) rate-induced tipping
    - Kuehn (2022) multiple time scales + bifurcations
"""

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.integrate import quad
import warnings


# ---------------------------------------------------------------------------
# 1. Master equation RHS (mean-field, 1D)
# ---------------------------------------------------------------------------

def rhs(A, alpha_s=0.06, C_mean=1.0, alpha_l=0.01, Pi=1.0,
        delta=0.01, f=0.0, A_c=1.0):
    """
    Mean-field 1D RHS with natural boundary enforcement.
    At A=0: only floor can drive it up; at A=1: only decay drives down.
    The multiplicative (1−A) and f·(1−A/A_c) terms handle this naturally,
    but clamp A to [1e-9, 1−1e-9] to prevent numerical blowup.
    """
    A = np.asarray(A, dtype=float)
    scalar = A.ndim == 0
    A = np.clip(A, 1e-9, 1 - 1e-9)
    beta = alpha_s * C_mean
    tsv  = beta * A**2 * (1 - A)
    fep  = alpha_l * Pi * A * (1 - A)
    floor_term = f * (1 - A / A_c)
    decay = delta * (1 - 0.3 * A)
    result = tsv + fep + floor_term - decay
    return float(result) if scalar else result


def rhs_deriv(A, **kwargs):
    """Numerical derivative of rhs w.r.t. A (Jacobian at fixed point)."""
    eps = 1e-6
    return (rhs(A + eps, **kwargs) - rhs(A - eps, **kwargs)) / (2 * eps)


# ---------------------------------------------------------------------------
# 2. Fixed points
# ---------------------------------------------------------------------------

def find_fixed_points(n_scan=2000, **kwargs):
    """
    Find all fixed points of dA/dτ = 0 in [0, 1] by sign-change scanning.

    Returns list of (A*, λ, type) sorted ascending.
    λ = eigenvalue of Jacobian = rhs'(A*).
    type: 'stable' (λ<0), 'unstable' (λ>0), 'absorbing' (A*≈0).
    """
    A_grid = np.linspace(0.0, 1.0, n_scan)
    vals   = rhs(A_grid, **kwargs)

    fps = []
    for i in range(len(A_grid) - 1):
        if vals[i] * vals[i+1] < 0:
            try:
                a_star = brentq(lambda a: rhs(a, **kwargs),
                                A_grid[i], A_grid[i+1], xtol=1e-10)
                lam = rhs_deriv(a_star, **kwargs)
                fps.append(a_star)
            except ValueError:
                pass

    # always include A=0 (absorbing death state)
    lam0 = rhs_deriv(0.0, **kwargs)
    result = []
    for a in sorted(set([round(x, 8) for x in fps])):
        lam = rhs_deriv(a, **kwargs)
        if abs(a) < 0.01:
            t = 'absorbing'
        elif lam > 0:
            t = 'unstable'
        else:
            t = 'stable'
        result.append({'A': a, 'lambda': lam, 'type': t})

    # add A=0 if not caught
    if not any(r['A'] < 0.01 for r in result):
        result.insert(0, {'A': 0.0, 'lambda': lam0, 'type': 'absorbing'})

    return result


def get_watershed(fps):
    """
    Return the unstable fixed point (true watershed / TippingPoint).
    This is A*_unstable — NOT the ad hoc 0.75.
    """
    unstable = [fp for fp in fps if fp['type'] == 'unstable']
    if not unstable:
        return None
    return min(unstable, key=lambda x: x['A'])


# ---------------------------------------------------------------------------
# 3. Bifurcation: saddle-node δ*
# ---------------------------------------------------------------------------

def delta_star(alpha_s=0.06, C_mean=1.0, alpha_l=0.01, Pi=1.0,
               delta=None, f=0.0, A_c=1.0, n_scan=1000):
    """
    Find δ* = saddle-node bifurcation point.

    At δ*: the unstable and upper-stable fixed points merge and annihilate.
    Condition: rhs(A) = 0  AND  rhs'(A) = 0  simultaneously.

    Algorithm: sweep δ, count fixed points. δ* = transition 3→1.

    Returns: (delta_star, A_saddle)
    """
    delta_grid = np.linspace(0.001, 0.05, n_scan)
    prev_n = None
    d_star = None
    A_sad  = None

    for d in delta_grid:
        fps = find_fixed_points(alpha_s=alpha_s, C_mean=C_mean,
                                alpha_l=alpha_l, Pi=Pi,
                                delta=d, f=f, A_c=A_c)
        n = sum(1 for fp in fps if fp['type'] in ('stable', 'unstable'))
        if prev_n is not None and prev_n >= 2 and n < 2:
            d_star = d
            # refine with bisection
            def count_nontrivial(dd):
                fps2 = find_fixed_points(alpha_s=alpha_s, C_mean=C_mean,
                                         alpha_l=alpha_l, Pi=Pi,
                                         delta=dd, f=f, A_c=A_c)
                return sum(1 for fp in fps2
                           if fp['type'] in ('stable', 'unstable'))
            # binary search for exact crossing
            lo, hi = delta_grid[np.where(delta_grid == d)[0][0] - 1], d
            for _ in range(50):
                mid = (lo + hi) / 2
                if count_nontrivial(mid) >= 2:
                    lo = mid
                else:
                    hi = mid
            d_star = (lo + hi) / 2
            # find A at saddle
            fps_at = find_fixed_points(alpha_s=alpha_s, C_mean=C_mean,
                                        alpha_l=alpha_l, Pi=Pi,
                                        delta=d_star * 0.9999, f=f, A_c=A_c)
            ws = get_watershed(fps_at)
            A_sad = ws['A'] if ws else None
            break
        prev_n = n

    return d_star, A_sad


def delta_star_vs_floor(floor_values, **base_kwargs):
    """
    Compute δ*(f) curve — the bifurcation boundary as function of floor.
    Verifies Finding 2 (floor expands survival region).

    Returns: array of (floor, delta_star, A_unstable)
    """
    results = []
    for f in floor_values:
        kw = dict(base_kwargs)
        kw['f'] = f
        ds, As = delta_star(**kw)
        results.append({'floor': f, 'delta_star': ds, 'A_unstable': As})
        print(f"  floor={f:.4f}  δ*={ds:.5f}  A*_unstable={As:.4f}")
    return results


# ---------------------------------------------------------------------------
# 4. Quasipotential V(A)
# ---------------------------------------------------------------------------

def quasipotential(A_range, **kwargs):
    """
    Compute quasipotential V(A) = -∫₀ᴬ rhs(a) da.

    For gradient systems: dA/dτ = -dV/dA → V(A) = -∫ rhs da.
    Fixed points = extrema of V:
        - stable attractors = local minima of V
        - unstable saddles  = local maxima of V

    Barrier height ΔV = V(A*_unstable) - V(A*_life):
        → Kramers escape rate ∝ exp(-ΔV / σ²)

    Returns: (A_array, V_array)
    """
    V = np.zeros_like(A_range)
    for i in range(1, len(A_range)):
        integrand, _ = quad(lambda a: -rhs(a, **kwargs),
                            A_range[i-1], A_range[i])
        V[i] = V[i-1] + integrand
    return A_range, V


def barrier_height(A_range=None, **kwargs):
    """
    Compute ΔV = V(A*_unstable) - V(A*_life).
    Higher ΔV → more resistant to noise (Kramers).

    Returns: {'delta_V': float, 'A_unstable': float, 'A_life': float,
              'V_unstable': float, 'V_life': float}
    """
    if A_range is None:
        A_range = np.linspace(0.01, 0.999, 5000)

    fps = find_fixed_points(**kwargs)
    ws  = get_watershed(fps)
    stable_upper = [fp for fp in fps if fp['type'] == 'stable']

    if ws is None or not stable_upper:
        return None

    A_life = max(stable_upper, key=lambda x: x['A'])['A']
    A_uns  = ws['A']

    _, V = quasipotential(A_range, **kwargs)

    # interpolate at fixed points
    V_uns  = float(np.interp(A_uns,  A_range, V))
    V_life = float(np.interp(A_life, A_range, V))

    return {
        'delta_V':    V_uns - V_life,   # barrier from life basin
        'A_unstable': A_uns,
        'A_life':     A_life,
        'V_unstable': V_uns,
        'V_life':     V_life,
    }


def kramers_escape_rate(sigma, **kwargs):
    """
    Kramers escape rate: Γ ≈ exp(-ΔV / σ²).
    Mean time to collapse: T_escape ≈ 1/Γ (in relational time units).

    sigma: noise amplitude in SDE  dA = rhs·dτ + σ·dW

    Returns: {'rate': float, 'mean_escape_time': float, 'delta_V': float}
    """
    bh = barrier_height(**kwargs)
    if bh is None:
        return None
    dV = bh['delta_V']
    rate = np.exp(-dV / (sigma**2 + 1e-30))
    return {
        'rate':             rate,
        'mean_escape_time': 1.0 / (rate + 1e-300),
        'delta_V':          dV,
        'sigma':            sigma,
    }


# ---------------------------------------------------------------------------
# 5. Quick diagnostics
# ---------------------------------------------------------------------------

def report(title="UAF 1D analysis", **kwargs):
    """Print full analytical report for given parameters."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  params: {kwargs}")
    print(f"{'='*60}")

    fps = find_fixed_points(**kwargs)
    print("\nFixed points:")
    for fp in fps:
        print(f"  A* = {fp['A']:.4f}  λ = {fp['lambda']:+.5f}  [{fp['type']}]")

    ws = get_watershed(fps)
    if ws:
        print(f"\nWatershed (true TippingPoint): A*_unstable = {ws['A']:.4f}")
    else:
        print("\nNo unstable fixed point — system in single-basin regime")

    bh = barrier_height(**kwargs)
    if bh:
        print(f"\nQuasipotential barrier:")
        print(f"  ΔV = {bh['delta_V']:.5f}")
        print(f"  V(A*_unstable) = {bh['V_unstable']:.5f}")
        print(f"  V(A*_life)     = {bh['V_life']:.5f}")
        for sigma in [0.01, 0.02, 0.05]:
            kr = kramers_escape_rate(sigma, **kwargs)
            print(f"  σ={sigma}: T_escape ≈ {kr['mean_escape_time']:.1e} steps")

    ds, As = delta_star(**kwargs)
    if ds:
        print(f"\nSaddle-node bifurcation: δ* = {ds:.5f}  (at A_saddle ≈ {As:.4f})")
        print(f"  Current δ = {kwargs.get('delta', 0.01):.5f}  "
              f"margin = {ds - kwargs.get('delta', 0.01):.5f}")
    print()


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default UAF v5 parameters
    BASE = dict(alpha_s=0.06, C_mean=1.0, alpha_l=0.01, Pi=1.0,
                delta=0.01, f=0.0, A_c=1.0)

    report("UAF v5 baseline", **BASE)

    # δ*(f) curve — verifies Finding 2
    print("δ*(floor) — bifurcation boundary expansion:")
    floors = [0.000, 0.001, 0.002, 0.005, 0.010]
    kw = dict(BASE)
    kw.pop('f')
    results = delta_star_vs_floor(floors, **kw)

    # Noise robustness at baseline
    print("\nKramers escape rates (baseline, varying noise):")
    for sigma in [0.005, 0.01, 0.02, 0.05]:
        kr = kramers_escape_rate(sigma, **BASE)
        if kr:
            print(f"  σ={sigma:.3f}: ΔV={kr['delta_V']:.4f}  "
                  f"T_escape={kr['mean_escape_time']:.2e}")
