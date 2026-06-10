"""
UAF v5.1 — NPG_net: Energy-Honest Normalised Performance Gain
==============================================================
Extends the existing NPG rule with floor energy cost.

Problem: a model can trivially improve NPG by increasing floor —
this pumps energy into the system without improving the mechanism.
NPG_net penalises floor cost, making survival "honest".

NPG (existing):
    NPG(M; D, B) = (L(B,D) − L(M,D)) / (L(B,D) + ε)

NPG_net (new):
    NPG_net = (F_base − F_model − λ · E_floor) / (F_base + ε)

where:
    F        = variational free energy = mean surprisal over trajectory
    E_floor  = Σ_τ Σ_i f_i(τ)² / (N · T)   [mean squared floor input]
    λ        = floor cost weight (default 1.0)

Interpretation:
    NPG_net > 0  → model genuinely better, even accounting for metabolic cost
    NPG_net ≤ 0  → reject or flag as "expensive rescue"
    NPG_net < NPG by large margin → survival was floor-dependent

References:
    - Friston (2019) variational free energy and ELBO
    - Dziugaite & Roy (2021) PAC-Bayes bounds
    - Stochastic thermodynamics: free energy ↔ work balance
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Free energy proxy
# ---------------------------------------------------------------------------

def free_energy_trajectory(A_traj, PE_traj=None, Pi_traj=None, eps=1e-10):
    """
    Compute variational free energy F along a trajectory.

    Minimal proxy (when full generative model unavailable):
        F_k = -log(A_k + ε) + 0.5 · Π_k · PE_k²

    Or just surprisal: F_k = -log(A_k + ε)   [baseline version]

    A_traj:  array (N_agents, T_steps) or (T_steps,) for 1D
    PE_traj: optional prediction errors, same shape
    Pi_traj: optional precisions, same shape

    Returns: F_mean (scalar) — mean free energy across agents and time.
    """
    A = np.atleast_2d(A_traj)         # (N, T)
    surprisal = -np.log(A + eps)

    if PE_traj is not None and Pi_traj is not None:
        PE = np.atleast_2d(PE_traj)
        Pi = np.atleast_2d(Pi_traj)
        F  = surprisal + 0.5 * Pi * PE**2
    else:
        F = surprisal

    return float(np.mean(F))


def free_energy_from_final(A_final, **kwargs):
    """
    Shorthand: F from final state distribution only.
    For single-basin comparison (fast NPG estimate).
    """
    return free_energy_trajectory(A_final, **kwargs)


# ---------------------------------------------------------------------------
# 2. Floor energy cost
# ---------------------------------------------------------------------------

def floor_energy(f_traj):
    """
    E_floor = mean(f_i(τ)²) over time and agents.

    f_traj: scalar (constant floor), or array (N, T) or (T,).
    """
    if np.isscalar(f_traj):
        return float(f_traj**2)
    return float(np.mean(np.asarray(f_traj)**2))


# ---------------------------------------------------------------------------
# 3. NPG and NPG_net
# ---------------------------------------------------------------------------

def npg(F_base, F_model, eps=1e-8):
    """
    Standard NPG: (F_base - F_model) / (F_base + ε).
    Positive = model better than baseline.
    Range: (-∞, 1].
    """
    return (F_base - F_model) / (F_base + eps)


def npg_net(F_base, F_model, f_cost, lam=1.0, eps=1e-8):
    """
    Energy-honest NPG:
        NPG_net = (F_base - F_model - λ·E_floor) / (F_base + ε)

    f_cost: scalar or array — floor input(s), passed to floor_energy()
    lam:    floor cost weight

    Returns dict with both NPG and NPG_net for comparison.
    """
    E  = floor_energy(f_cost)
    ng = npg(F_base, F_model, eps)
    ng_net = (F_base - F_model - lam * E) / (F_base + eps)

    return {
        'NPG':        ng,
        'NPG_net':    ng_net,
        'E_floor':    E,
        'floor_cost': lam * E,
        'verdict':    _verdict(ng, ng_net),
    }


def _verdict(ng, ng_net):
    if ng_net > 0:
        return 'ACCEPT: genuinely better'
    elif ng > 0 and ng_net <= 0:
        return 'EXPENSIVE RESCUE: survival floor-dependent'
    else:
        return 'REJECT: worse than baseline'


# ---------------------------------------------------------------------------
# 4. Batch comparison (sweep floor values)
# ---------------------------------------------------------------------------

def compare_floor_sweep(results, F_base, lam=1.0, eps=1e-8):
    """
    Compare multiple model runs that differ in floor.

    results: list of dicts with keys:
        'floor', 'F_model', 'survived', and optionally 'f_traj'

    Returns: list of results augmented with NPG, NPG_net, verdict.
    """
    output = []
    for r in results:
        f_cost = r.get('f_traj', r['floor'])
        metrics = npg_net(F_base, r['F_model'], f_cost, lam=lam, eps=eps)
        row = dict(r, **metrics)
        output.append(row)
    return output


# ---------------------------------------------------------------------------
# 5. Optimal floor
# ---------------------------------------------------------------------------

def find_optimal_floor(F_base, F_model_fn, floor_range,
                       lam=1.0, eps=1e-8):
    """
    Find floor* = argmax NPG_net(f).
    Beyond floor*, added metabolic cost outweighs free energy gain.

    F_model_fn: callable f → F_model (e.g., from simulation or analytics)
    floor_range: array of floor values to test

    Returns: {'floor_star', 'NPG_net_star', 'floor_range', 'NPG_net_curve'}
    """
    npg_net_curve = []
    for f in floor_range:
        F_m = F_model_fn(f)
        r   = npg_net(F_base, F_m, f, lam=lam, eps=eps)
        npg_net_curve.append(r['NPG_net'])

    npg_net_curve = np.array(npg_net_curve)
    idx_star      = np.argmax(npg_net_curve)

    return {
        'floor_star':       floor_range[idx_star],
        'NPG_net_star':     npg_net_curve[idx_star],
        'floor_range':      floor_range,
        'NPG_net_curve':    npg_net_curve,
    }


# ---------------------------------------------------------------------------
# 6. Pretty report
# ---------------------------------------------------------------------------

def report(F_base, F_model, floor, lam=1.0, label=""):
    r = npg_net(F_base, F_model, floor, lam=lam)
    print(f"\nNPG report {label}")
    print(f"  F_base   = {F_base:.4f}")
    print(f"  F_model  = {F_model:.4f}")
    print(f"  E_floor  = {r['E_floor']:.6f}  (λ·E = {r['floor_cost']:.4f})")
    print(f"  NPG      = {r['NPG']:+.4f}")
    print(f"  NPG_net  = {r['NPG_net']:+.4f}")
    print(f"  Verdict  : {r['verdict']}")
    return r


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Baseline: single agent, dies
    A_base = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
    F_base = free_energy_trajectory(A_base)

    # Model 1: survives with small floor
    A_m1 = np.array([0.25, 0.35, 0.55, 0.70, 0.80])
    F_m1 = free_energy_trajectory(A_m1)
    report(F_base, F_m1, floor=0.001, lam=1.0, label="small floor")

    # Model 2: survives but with large floor
    A_m2 = np.array([0.30, 0.50, 0.75, 0.80, 0.82])
    F_m2 = free_energy_trajectory(A_m2)
    report(F_base, F_m2, floor=0.05, lam=1.0, label="large floor")

    # Optimal floor sweep (analytical proxy)
    floors = np.linspace(0.0, 0.05, 50)
    # synthetic: F_model(f) = F_base * exp(-10*f)  [floor helps, diminishing returns]
    F_model_fn = lambda f: F_base * np.exp(-10 * f)
    opt = find_optimal_floor(F_base, F_model_fn, floors, lam=1.0)
    print(f"\nOptimal floor: f* = {opt['floor_star']:.4f}  "
          f"NPG_net* = {opt['NPG_net_star']:.4f}")
