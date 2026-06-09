# =============================================================================
# EXP-UAF-GL-001 | UAF+GL HYBRID: TDGL + Particle-Tracker
# Status: VERIFIED ✓
# Date: June 2026
# Author: 131ym / UAF v5.1
# =============================================================================
"""
EXPERIMENT: UAF+GL Hybrid Model — Abrikosov Vortices with Magnus+Hall Separation

HYPOTHESIS:
    UAF adaptive dynamics (A_i update rule) maps onto TDGL order parameter ψ.
    Vortex charges ±1 in type-II superconductor separate under:
      - Magnus force: F = q*(J × B̂)  → separation along x
      - Hall effect:  (1+iη)*dψ/dt   → separation along y

METHOD:
    Two coupled layers:
    1. TDGL field:     (1+iη)*dψ/dt = -(α+β|ψ|²)ψ + γ*D²ψ + noise
       - Landau gauge: D_y = ∂_y - iHx
       - Open boundaries: vortex nucleation (y=0), absorption (y=L)
    2. Particle tracker: vortices as point particles
       - v_i = μ*(F_Magnus + F_vv + F_pin) + noise
       - F_Magnus = q_i * B_z * (-J_y, J_x)
       - Back-coupling: core_mask suppresses |ψ| at vortex positions

PARAMETERS:
    α=-0.55, β=0.55, γ=0.35, H=0.45, η=0.50
    J_transport=(0.80, 0), B_z=0.45
    N_vortex=44 (22+/22-), L=96, dt=0.006

RESULTS (25k steps):
    TDGL:
        mean|ψ| = 0.923  (theory: 1.0)   STATUS: ✓ stable
        std|ψ|  = 0.156                   STATUS: ✓ heterogeneous
        N_vortex = 38-44                  STATUS: ✓ stable flux flow

    Charge separation:
        Δȳ(Hall)     = 30.6 px            STATUS: ✓✓ VERIFIED
        |Δx̄|(Magnus) =  7.7 px            STATUS: △ present, weak
        Trend Δȳ: +0.84 px/1000 steps
        Saturation expected: ~50k steps (Δȳ → L/2 = 48px)

UAF INTERPRETATION:
    - |ψ|² ↔ A_i (local activation/condensate density)
    - std|ψ| ↔ heterogeneity in UAF network
    - Vortex cores: A_i → 0 (local suppression = prediction error spike)
    - Hall separation: topological charge conservation under asymmetric dynamics
    - Maps to UAF bistability (EXP-028): vortex = saddle-node in phase space

OPEN QUESTIONS → next experiments:
    Q1: Does vortex density scale as N_v ~ H·L²/(2π)?  [→ EXP-UAF-GL-002]
    Q2: Does Δȳ(t) follow power law or exponential?   [→ EXP-UAF-GL-003]
    Q3: Connection to SIS epidemic model on vortex lattice? [links to Q5]

FILES:
    uaf_gl_hybrid_fast.py   — full simulation code (vectorized)
    uaf_gl_hybrid_30k.png   — results plot
"""

# ── Key parameters ─────────────────────────────────────────────────────────
PARAMS = dict(
    alpha=-0.55, beta=0.55, gamma=0.35,
    H=0.45, eta=0.50,
    J_transport=(0.80, 0.0), B_z=0.45,
    N_pos=22, N_neg=22, L=96,
    dt=0.006, steps_run=25000, noise=0.005,
    mob=0.40, vv_str=0.6, vv_rng=7.0, xi_core=3.0,
)

# ── Verified findings ───────────────────────────────────────────────────────
FINDINGS = {
    'mean_psi_final':  0.923,
    'std_psi_final':   0.156,
    'N_vortex_stable': 40,
    'delta_y_hall':    30.6,   # px, at 25k steps
    'delta_x_magnus':  7.7,    # px, at 25k steps
    'trend_dy_per_kstep': 0.84,
    'saturation_steps_est': 50000,
    'status': 'VERIFIED',
}

# ── UAF mapping ─────────────────────────────────────────────────────────────
UAF_MAPPING = {
    '|psi|^2':     'A_i — local condensate = UAF activation',
    'std|psi|':    'heterogeneity — UAF network disorder',
    'vortex_core': 'A_i → 0 — prediction error spike / bifurcation point',
    'hall_sep':    'topological charge drift — asymmetric UAF update',
    'flux_flow':   'directed vortex transport = UAF signal propagation',
}


# ── findings_v5.py entry ────────────────────────────────────────────────────
FINDINGS_V5_ENTRY = """
# EXP-UAF-GL-001 | June 2026
exp_uaf_gl_001 = Finding(
    id='EXP-UAF-GL-001',
    title='UAF+GL Hybrid: TDGL + Particle-Tracker Magnus+Hall vortex separation',
    status='VERIFIED',
    params=dict(alpha=-0.55, beta=0.55, gamma=0.35, H=0.45, eta=0.50,
                J_tr=0.80, N_vortex=44, L=96, steps=25000),
    results=dict(
        mean_psi=0.923, std_psi=0.156, N_stable=40,
        delta_y_hall=30.6, delta_x_magnus=7.7,
        trend_dy='0.84 px/kstep', saturation='~50k steps',
    ),
    uaf_mapping=dict(
        psi_sq='A_i condensate density',
        vortex_core='prediction error spike / A_i→0',
        hall_sep='topological charge under asymmetric dynamics',
    ),
    open_questions=[
        'Q_GL1: N_v ~ H*L^2/(2pi)?',
        'Q_GL2: Δy(t) power law or exponential?',
        'Q_GL3: link to SIS/vortex lattice (UAF Q5)',
    ],
    files=['uaf_gl_hybrid_v5.1.py', 'uaf_gl_hybrid_30k.png'],
)
"""
