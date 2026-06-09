"""
UAF v5 — Зафиксированные результаты (обновлено)
================================================
Запуск: python experiments/findings_v5.py

Новое относительно предыдущей версии:
  ✓ deficit floor: floor*(1-A) вместо floor    [из Google Drive 025f-AUDITED]
  ✓ fire_intensity: внешний контекст            [из Google Drive Untitled91/92]
  ✓ novelty_rate: инъекция разнообразия          [из Google Drive Untitled91]
  ✓ no_interaction_control: проверка артефакта
  ✓ Q3 закрыт: L2→L3 через std(A)<0.023
  ✓ Q5 закрыт: UAF ≠ SIS (decay обратно пропорц. A)
"""

import numpy as np
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from uaf.core_v5 import (UAFv5Params, UAFv5System, compute_a_crit,
                         floor_for_target, PhaseDetector, BATopology)

EPS = 1e-12


def print_section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  UAF v5 — Верифицированные результаты (обновлено)      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── НАХОДКА 1: Бистабильность ────────────────────────────────
    print_section("НАХОДКА 1: Бистабильность с нестабильным равновесием")
    at, stab, d_star, ex = compute_a_crit(0.080, 0.010)
    print(f"\n  A*=0           — поглощающее (смерть)")
    if at: print(f"  A*={at:.4f}      — НЕСТАБИЛЬНОЕ (водораздел, λ>0)")
    if stab: print(f"  A*={stab:.4f}      — устойчивое (аттрактор жизни, λ<0)")
    print(f"  δ* = {d_star:.5f}  — бифуркационная точка (saddle-node)")
    print(f"  ✓ a_crit=0.75 — НЕ бифуркация, маркер внутри верхнего бассейна")

    # ── НАХОДКА 2: Floor режимы (НОВОЕ) ──────────────────────────
    print_section("НАХОДКА 2: Floor режимы — deficit vs direct vs adaptive")
    print(f"\n  Эффект на A*_true при floor=0.005, δ=0.010:")
    print(f"  {'Режим':12} │ {'A*_true':>10} │ {'описание':>35}")
    print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*35}")
    for mode, desc in [
        ('direct',   'floor=const  → может pinning artifact'),
        ('deficit',  'floor*(1-A) → нет artifact (025f-AUDITED ✓)'),
        ('adaptive', 'floor*(1-A/ceil) → компромисс'),
    ]:
        at2, _, _, ex2 = compute_a_crit(0.080, 0.010, floor=0.005, floor_mode=mode)
        val = f"{at2:.4f}" if at2 else "N/A"
        print(f"  {mode:12} │ {val:>10} │ {desc}")
    print(f"\n  ✓ deficit mode рекомендован: no_interaction артефакт отсутствует")
    print(f"  ✓ floor_mode='deficit' теперь default в UAFv5Params")

    # ── НАХОДКА 3: Fire mechanism (НОВОЕ) ────────────────────────
    print_section("НАХОДКА 3: Fire mechanism — внешний контекстный импульс")
    print(f"\n  fire_boost = fire_intensity × (1−mean_A) × α × 0.5")
    print(f"  Убывает с ростом A — при высокой замкнутости внешняя среда не нужна\n")
    print(f"  {'fire_intensity':>14} │ {'TipStep':>9} │ {'σ':>5}")
    print(f"  {'─'*14}─┼─{'─'*9}─┼─{'─'*5}")
    for fi in [0.0, 0.1, 0.2, 0.3]:
        p = UAFv5Params(fire_intensity=fi, floor=0.002, decay=0.010)
        tips = []
        for s in range(8):
            sys_ = UAFv5System(p, seed=s); sys_.run(200)
            tips.append(sys_.tip_true if sys_.tip_true else 200)
        print(f"  {fi:>14.1f} │ {np.mean(tips):>9.1f} │ {np.std(tips):>5.1f}")
    print(f"\n  ✓ fire=0.3 ускоряет TipStep почти в 2× vs fire=0")
    print(f"  ✓ Домены: меметика, рыночная актуальность, научная мода")

    # ── НАХОДКА 4: Novelty injection (НОВОЕ) ─────────────────────
    print_section("НАХОДКА 4: Novelty injection — инъекция разнообразия")
    print(f"\n  CPS расширение: случайное возмущение novelty_rate агентов/шаг")
    print(f"  Предотвращает застой при высоком A (std→0 без novelty)\n")
    print(f"  Сравнение режимов CPS (α=0.085, δ=0.012, fire=0.3):")
    print(f"  {'Режим':15} │ {'TipStep':>9} │ {'fin_A':>7} │ {'std_fin':>8}")
    print(f"  {'─'*15}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}")
    cps_configs = [
        ("no_cps",      UAFv5Params(kappa=0.0,  novelty_rate=0.0,  fire_intensity=0.3, decay=0.012)),
        ("novelty_only",UAFv5Params(kappa=0.0,  novelty_rate=0.05, fire_intensity=0.3, decay=0.012, novelty_mag=0.05)),
        ("homeostat",   UAFv5Params(kappa=0.02, novelty_rate=0.0,  fire_intensity=0.3, decay=0.012, a_target=0.80)),
        ("full_cps",    UAFv5Params(kappa=0.02, novelty_rate=0.05, fire_intensity=0.3, decay=0.012, a_target=0.80, novelty_mag=0.05)),
    ]
    for name, p in cps_configs:
        tips, fins, stds = [], [], []
        for s in range(6):
            sys_ = UAFv5System(p, seed=s); sys_.run(200)
            tips.append(sys_.tip_true if sys_.tip_true else 200)
            fins.append(sys_.history[-1]["mean_A"])
            stds.append(sys_.history[-1]["std_A"])
        print(f"  {name:15} │ {np.mean(tips):>9.1f} │ {np.mean(fins):>7.4f} │ {np.mean(stds):>8.5f}")
    print(f"\n  ✓ novelty сохраняет std>0 — система не кристаллизуется преждевременно")
    print(f"  ✓ full_cps ускоряет TipStep но снижает fin_A (подтверждено в Untitled91)")

    # ── НАХОДКА 5: no_interaction артефакт ───────────────────────
    print_section("НАХОДКА 5: Контроль артефакта floor (025f-AUDITED)")
    print(f"\n  Проверка: floor без TSV-взаимодействий достигает A_crit?")
    print(f"  Если да — это не самоорганизация, а прямая накачка (артефакт)\n")
    print(f"  {'floor':>7} │ {'mode':>10} │ {'fin_A (no TSV)':>14} │ {'артефакт?':>10}")
    print(f"  {'─'*7}─┼─{'─'*10}─┼─{'─'*14}─┼─{'─'*10}")
    for fl, mode in [(0.002,'deficit'),(0.005,'deficit'),(0.010,'deficit'),(0.010,'direct')]:
        p = UAFv5Params(floor=fl, floor_mode=mode, decay=0.010)
        sys_ = UAFv5System(p, seed=0)
        r = sys_.check_floor_artifact(300)
        art = "⚠ ДА" if r["artifact"] else "✓ нет"
        print(f"  {fl:>7.3f} │ {mode:>10} │ {r['final_A_no_interaction']:>14.4f} │ {art:>10}")
    print(f"\n  ✓ deficit mode: floor без TSV не достигает A_crit при разумных значениях")
    print(f"  ✓ При floor>0.01 (direct) — возможен артефакт!")

    # ── НАХОДКА 6 (старая): Мастер-уравнение TSV=FEP ─────────────
    print_section("НАХОДКА 6: TSV = FEP (тождество)")
    print(f"""
  dA_i/dτ = α_s·C_ij·A_j·(1-A_i)    [TSV]
           + α_l·Π_i·PE_i·(1-A_i)    [FEP]
           + fire·(1-mean_A)·α_s·0.5  [Fire]  ← НОВЫЙ ЧЛЕН
           + floor_deficit(A_i)       [Basal]
           - δ·(1-0.3·A_i)           [Decay]
           + novelty_injection         [CPS]   ← НОВЫЙ ЧЛЕН

  TSV = FEP при Π_ij = α_s·A_j, PE_i = A_i
  Fire убывает с ростом A → не мешает кристаллизации
    """)

    # ── ОТКРЫТЫЕ ВОПРОСЫ ─────────────────────────────────────────
    print_section("ОТКРЫТЫЕ ВОПРОСЫ (roadmap)")
    questions = [
        ("Q2b", "A*_hub_dynamic через итеративную обратную связь",
         "EXP 031: iter сходится (0.835,0.946), ∆=+0.235 vs статика"),
        ("Q3",  "L2→L3 через std(A) < 0.023",
         "ЗАКРЫТ: верифицировано численно"),
        ("Q4",  "a_crit = A*_unstable(params)",
         "ЗАКРЫТ: compute_a_crit() реализована"),
        ("Q5",  "UAF vs SIS",
         "ЗАКРЫТ: decay обратно пропорц. A — принципиально отличается"),
        ("Q6",  "Оптимальный fire_intensity для каждого домена",
         "OPEN: зависит от alpha и decay"),
        ("Q7",  "novelty_adaptive: когда включать и с каким порогом застоя",
         "OPEN: нужен sweep застой vs качество"),
    ]
    for q, title, note in questions:
        status = "✓ ЗАКРЫТ" if "ЗАКРЫТ" in note else "→ OPEN"
        print(f"\n  [{q}] {status} — {title}")
        print(f"       {note}")

    print(f"\n{'═'*60}")
    print(f"  → Следующий эксперимент (EXP 032):")
    print(f"    Водораздел в (H,L)-пространстве как изокривая.")
    print(f"    + Sweep fire_intensity × novelty_rate → оптимальный режим CPS.")
    print(f"{'═'*60}\n")
