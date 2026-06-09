"""
UAF v5 — uaf/core.py (обновлённый)
Новые механизмы из Google Drive:
  1. deficit-limited floor: floor*(1-A)   [025f-AUDITED]
  2. fire_intensity: внешний контекст     [Untitled91/92]
  3. novelty_rate: инъекция разнообразия  [Untitled91]
  4. no_interaction_control               [025f-AUDITED]
  5. floor_mode: direct|deficit|adaptive
  6. Q3 закрыт: L2→L3 через std(A)<0.023
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from scipy.optimize import brentq

EPS = 1e-12


@dataclass
class UAFv5Params:
    # TSV
    alpha_social:   float = 0.080
    alpha_learn:    float = 0.050
    alpha_cat:      float = 0.650
    epsilon_cost:   float = 0.020
    # Decay
    decay:          float = 0.010
    decay_mod:      float = 0.300
    # Basal floor (NEW: floor_mode)
    floor:          float = 0.002
    floor_mode:     str   = 'deficit'  # 'direct'|'deficit'|'adaptive'
    a_ceiling:      float = 0.950
    # Fire (NEW: внешний контекстный импульс)
    fire_intensity: float = 0.000   # 0=выкл
    # CPS расширенный (NEW: novelty)
    kappa:          float = 0.000
    a_target:       float = 0.800
    kappa_zone:     float = 0.050
    novelty_rate:   float = 0.000   # 0=выкл
    novelty_mag:    float = 0.050
    novelty_adaptive: bool = False
    # Структура
    n_agents:       int   = 60
    density:        float = 0.250
    K:              int   = 3
    # TippingPoint
    a_crit:         float = 0.750
    # Начальные условия
    a_init_low:     float = 0.250
    a_init_high:    float = 0.350


def compute_a_crit(alpha=0.080, delta=0.010, epsilon=0.02,
                   floor=0.0, a_ceil=0.95, floor_mode='deficit'):
    beta = alpha - epsilon
    def floor_eff(a):
        if floor <= 0: return 0.0
        if floor_mode == 'direct': return floor
        elif floor_mode == 'deficit': return floor * (1.0 - a)
        else: return floor * (1.0 - a / a_ceil)
    def f(a): return beta*a**2*(1-a) - delta*(1-0.3*a) + floor_eff(a)
    def n_roots(d):
        def g(a): return beta*a**2*(1-a) - d*(1-0.3*a)
        fv = np.array([g(a) for a in np.linspace(1e-4,1-1e-4,100_000)])
        return len(np.where(np.diff(np.sign(fv)))[0])
    lo, hi = 1e-4, 0.1
    for _ in range(50):
        mid=(lo+hi)/2
        if n_roots(mid)>=2: lo=mid
        else: hi=mid
    delta_star=(lo+hi)/2
    A_scan=np.linspace(1e-4,1-1e-4,500_000)
    fv=np.array([f(a) for a in A_scan])
    crossings=np.where(np.diff(np.sign(fv)))[0]
    roots=[]
    for c in crossings:
        try:
            r=brentq(f,A_scan[c],A_scan[c+1],xtol=1e-10)
            lam=beta*(2*r-3*r**2)+0.3*delta
            roots.append((r,lam))
        except: pass
    if len(roots)<2: return None,None,delta_star,False
    return min(roots,key=lambda x:x[0])[0], max(roots,key=lambda x:x[0])[0], delta_star, True


def floor_for_target(a_target, alpha=0.080, delta=0.010, epsilon=0.02,
                     a_ceil=0.95, floor_mode='deficit'):
    def cost(fl):
        at,_,_,ex=compute_a_crit(alpha,delta,epsilon,float(fl),a_ceil,floor_mode)
        return (at-a_target) if (ex and at) else -a_target
    if cost(0.0)<0: return None
    fl_max=0.0
    for fl_test in np.linspace(0.001,0.020,100):
        _,_,_,ex=compute_a_crit(alpha,delta,epsilon,fl_test,a_ceil,floor_mode)
        if not ex: fl_max=fl_test; break
    if fl_max==0: fl_max=0.012
    try: return brentq(cost,0.0,fl_max-1e-5,xtol=1e-8)
    except: return None


class BATopology:
    def __init__(self, n, m=3, eta=0.65, seed=0):
        rng=np.random.default_rng(seed)
        self.n=n; self.adj=[[] for _ in range(n)]; m=min(m,n-1)
        for i in range(m+1):
            for j in range(i+1,m+1): self.adj[i].append(j); self.adj[j].append(i)
        degs=np.array([len(self.adj[i]) for i in range(n)],dtype=float)
        for v in range(m+1,n):
            total=degs[:v].sum(); p=degs[:v]/total if total>0 else np.ones(v)/v
            chosen=set()
            while len(chosen)<min(m,v): chosen.add(int(rng.choice(v,p=p)))
            for nb in chosen:
                self.adj[v].append(nb); self.adj[nb].append(v)
                degs[v]+=1; degs[nb]+=1
        for i in range(n):
            if not self.adj[i]:
                j=(i+1)%n; self.adj[i].append(j); self.adj[j].append(i)
        self.degrees=np.array([len(self.adj[i]) for i in range(n)],dtype=float)
        self.catalysis=np.clip((self.degrees/max(self.degrees.mean(),EPS))**eta,0.5,3.5)
        self.n_hubs=int(np.sum(self.degrees>self.degrees.mean()+self.degrees.std()))


class PhaseDetector:
    L3_STD_THRESHOLD = 0.023  # Q3 верифицирован: L2→L3 при std<0.023
    @staticmethod
    def jacobian(A_i, A_j_mean, p):
        beta=p.alpha_social-p.epsilon_cost
        return beta*A_j_mean*(1-2*A_i) - p.decay*(1-0.3*A_i)
    @classmethod
    def phase(cls, A, p):
        mA=float(np.mean(A)); sA=float(np.std(A))
        lam=float(np.mean([cls.jacobian(float(a),mA,p) for a in A]))
        if lam>0.01: ph='chaos'
        elif lam>-0.005: ph='transition'
        elif sA>cls.L3_STD_THRESHOLD: ph='coherent'
        else: ph='integrated'
        return ph, lam
    @staticmethod
    def label(ph):
        return {'chaos':'L0-хаос','transition':'L1-переход',
                'coherent':'L2-когерент','integrated':'L3-интеграция'}.get(ph,'?')


def uaf_step(A, topo, precision, p, rng, step):
    N=len(A); dA=np.zeros(N); mean_A=float(np.mean(A))
    n_pairs=max(1,N//2); idx=rng.permutation(N)
    for pp in range(n_pairs):
        i=int(idx[(2*pp)%N]); j=int(idx[(2*pp+1)%N])
        Ai,Aj=A[i],A[j]
        ae=p.alpha_social*topo.catalysis[j]
        dA[i]+=ae*Aj*(1-Ai)-p.epsilon_cost*Ai*Aj*(1-Aj)
        dA[j]+=p.alpha_social*topo.catalysis[i]*Ai*(1-Aj)-p.epsilon_cost*Aj*Ai*(1-Ai)
        pe_i=max(0.0,Aj-Ai); pe_j=max(0.0,Ai-Aj)
        dA[i]+=p.alpha_learn*precision[i]*pe_i*(1-Ai)
        dA[j]+=p.alpha_learn*precision[j]*pe_j*(1-Aj)
    # Fire
    if p.fire_intensity>0:
        dA+=p.fire_intensity*(1-mean_A)*p.alpha_social*0.5
        dA+=rng.normal(0,p.fire_intensity*0.02,N)
    # Decay
    dA-=p.decay*(1.0-p.decay_mod*A)
    # Floor (три режима)
    if p.floor>0:
        if p.floor_mode=='direct':
            dA+=p.floor
        elif p.floor_mode=='deficit':
            dA+=p.floor*(1.0-A)           # 025f-AUDITED ✓
        else:
            dA+=p.floor*np.maximum(0.0,1.0-A/p.a_ceiling)
    # CPS гомеостаз
    if p.kappa>0:
        dev=A-p.a_target; mask=np.abs(dev)>p.kappa_zone
        dA-=p.kappa*dev*mask
    # CPS novelty
    if p.novelty_rate>0:
        n_p=max(1,int(p.novelty_rate*N))
        idx_p=rng.choice(N,size=n_p,replace=False)
        mag=p.novelty_mag
        if p.novelty_adaptive:
            stag=float(np.mean(np.abs(dA))<1e-4)
            mag*=(1.0+2.0*stag)
        dA[idx_p]+=rng.uniform(-mag,mag,n_p)
    # Precision update
    pe_g=np.abs(A-mean_A)
    prec_new=np.clip(precision+0.01*(0.5-pe_g),0.1,3.0)
    A_new=np.clip(A+dA,0.0,p.a_ceiling)
    ph,lam=PhaseDetector.phase(A_new,p)
    return A_new, prec_new, {
        "mean_A":float(np.mean(A_new)), "std_A":float(np.std(A_new)),
        "phase":ph, "lambda":lam, "mean_prec":float(np.mean(prec_new)),
        "fire_eff":float(p.fire_intensity*(1-mean_A)*p.alpha_social*0.5) if p.fire_intensity>0 else 0.0,
    }


class UAFv5System:
    def __init__(self, params=None, seed=42, interactions=True):
        self.p=params or UAFv5Params()
        self.interactions=interactions
        self.rng=np.random.default_rng(seed)
        self.topo=BATopology(self.p.n_agents,self.p.K,self.p.alpha_cat,seed=seed)
        self.A=self.rng.uniform(self.p.a_init_low,self.p.a_init_high,self.p.n_agents)
        self.precision=np.ones(self.p.n_agents)
        self.history: List[Dict]=[]
        self.tip_old=None; self.tip_true=None
        self.a_crit_true,self.a_stable,self.delta_star,self.bistable=\
            compute_a_crit(self.p.alpha_social,self.p.decay,self.p.epsilon_cost,
                           self.p.floor,self.p.a_ceiling,self.p.floor_mode)

    def step(self, t):
        if self.interactions:
            self.A,self.precision,m=uaf_step(self.A,self.topo,self.precision,self.p,self.rng,t)
        else:
            dA=np.zeros(self.p.n_agents)
            dA-=self.p.decay*(1.0-self.p.decay_mod*self.A)
            if self.p.floor>0:
                if self.p.floor_mode=='deficit': dA+=self.p.floor*(1.0-self.A)
                elif self.p.floor_mode=='direct': dA+=self.p.floor
                else: dA+=self.p.floor*np.maximum(0.0,1.0-self.A/self.p.a_ceiling)
            self.A=np.clip(self.A+dA,0.0,self.p.a_ceiling)
            ph,lam=PhaseDetector.phase(self.A,self.p)
            m={"mean_A":float(np.mean(self.A)),"std_A":float(np.std(self.A)),
               "phase":ph,"lambda":lam,"mean_prec":1.0,"fire_eff":0.0}
        m["step"]=t; m["event"]=""
        if self.tip_old is None and m["mean_A"]>=self.p.a_crit: self.tip_old=t
        if (self.tip_true is None and self.bistable and self.a_crit_true
                and m["mean_A"]>=self.a_crit_true):
            self.tip_true=t; m["event"]="TIPPING_POINT_TRUE"
        self.history.append(m); return m

    def run(self, steps=500, verbose_every=0):
        for t in range(steps):
            m=self.step(t)
            if verbose_every>0 and (t%verbose_every==0 or m["event"]):
                tipped=int(np.sum(self.A>=self.p.a_crit))
                ni="" if self.interactions else " [no-TSV]"
                print(f"  t={t:4d}: A={m['mean_A']:.4f} ±{m['std_A']:.4f} "
                      f"[{PhaseDetector.label(m['phase'])}] λ={m['lambda']:+.4f} "
                      f"tipped={tipped}/{self.p.n_agents}{ni}"
                      +(f"  ← {m['event']}" if m['event'] else ""))
        return self.history

    def check_floor_artifact(self, steps=500):
        """no_interaction_control: floor без TSV достигает A_crit?"""
        ctrl=UAFv5System(UAFv5Params(**self.p.__dict__),seed=0,interactions=False)
        ctrl.run(steps)
        fin=ctrl.history[-1]["mean_A"]
        artifact=fin>=self.p.a_crit
        return {"artifact":artifact,"final_A_no_interaction":fin,
                "message":"⚠ АРТЕФАКТ: floor без TSV достигает A_crit!" if artifact
                           else "✓ Артефакта нет: floor без TSV не достигает A_crit."}

    def npg(self, baseline_history):
        my=self.mean_A_history()
        surv=self.history[-1]["mean_A"]>=self.p.a_crit if self.history else False
        b_surv=baseline_history[-1]>=self.p.a_crit if baseline_history else False
        F_m=-(np.mean(my[-50:] if len(my)>=50 else my)+np.log(float(surv)+EPS))
        F_b=-(np.mean(baseline_history[-50:] if len(baseline_history)>=50 else baseline_history)
              +np.log(float(b_surv)+EPS))
        npg_v5=float((F_b-F_m)/(abs(F_b)+EPS))
        return {"NPG_v5":npg_v5,"F_model":F_m,"F_base":F_b,
                "survived":surv,"tip_old":self.tip_old,"tip_true":self.tip_true,
                "final_A":self.history[-1]["mean_A"] if self.history else 0.0,
                "a_crit_true":self.a_crit_true,"a_stable":self.a_stable}

    def mean_A_history(self): return [m["mean_A"] for m in self.history]
    def phase_trajectory(self): return [m["phase"] for m in self.history]
