#!/usr/bin/env python3
"""UAF+GL HYBRID — векторизованный, 30k шагов"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L = 96
rng = np.random.default_rng(314)

alpha=-0.55; beta=0.55; gamma=0.35; H=0.45; eta=0.50
dt=0.006; steps=30000; noise=0.005
N_pos=22; N_neg=22
J_tr=np.array([0.80,0.0]); B_z=H
mob=0.40; noise_v=0.06
vv_str=0.6; vv_rng=7.0; xi_core=3.0

A_eq=np.sqrt(-alpha/beta)
xc=(np.arange(L)/L-0.5)[np.newaxis,:]
print(f"|ψ|₀={A_eq:.3f}  steps={steps}")

theta_init=rng.uniform(-np.pi,np.pi,(L,L))
psi_r=A_eq*0.80*np.cos(theta_init)
psi_i=A_eq*0.80*np.sin(theta_init)

pin_f=np.zeros((L,L))
sites=rng.choice(L*L,size=int(0.06*L*L),replace=False)
pin_f.flat[sites]=rng.uniform(0.08,0.20,len(sites))

vpos=rng.uniform(8,L-8,(N_pos+N_neg,2))
vcharge=np.array([1.0]*N_pos+[-1.0]*N_neg); rng.shuffle(vcharge)

Y_grid,X_grid=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')

history={'mean_A':[],'std_A':[],'x_pos':[],'x_neg':[],
         'y_pos':[],'y_neg':[],'sep_x':[],'sep_y':[],'n_v':[]}
snapshots={}

def lap_open(f):
    r=np.roll(f,1,1)+np.roll(f,-1,1)-2*f
    r[1:-1,:]+=f[:-2,:]+f[2:,:]-2*f[1:-1,:]
    r[0,:]+=f[1,:]-f[0,:]; r[-1,:]+=f[-2,:]-f[-1,:]
    return r

def apply_bc(pr,pi,rng,noise):
    pr[0,:]*=0.10; pi[0,:]*=0.10
    pr[0,:]+=rng.normal(0,noise*3,L); pi[0,:]+=rng.normal(0,noise*3,L)
    pr[-1,:]*=0.10; pi[-1,:]*=0.10
    return pr,pi

def core_mask_vec(vpos,L,xi):
    """Векторизованная маска ядер"""
    if len(vpos)==0: return np.zeros((L,L))
    px=vpos[:,0][np.newaxis,np.newaxis,:]  # (1,1,N)
    py=vpos[:,1][np.newaxis,np.newaxis,:]
    dx=X_grid[:,:,np.newaxis]-px
    dy=Y_grid[:,:,np.newaxis]-py
    dx=dx-L*np.round(dx/L); dy=dy-L*np.round(dy/L)
    r2=dx**2+dy**2
    return np.clip(np.sum(np.exp(-r2/xi**2),axis=2),0,1)

def supercurrent_vec(psi_r,psi_i,vpos,L):
    """Векторизованный суперток во всех точках сразу"""
    ix=(vpos[:,0].astype(int))%L
    iy=(vpos[:,1].astype(int))%L
    dpr_dx=(psi_r[iy,(ix+1)%L]-psi_r[iy,(ix-1)%L])/2
    dpi_dx=(psi_i[iy,(ix+1)%L]-psi_i[iy,(ix-1)%L])/2
    dpr_dy=(psi_r[(iy+1)%L,ix]-psi_r[(iy-1)%L,ix])/2
    dpi_dy=(psi_i[(iy+1)%L,ix]-psi_i[(iy-1)%L,ix])/2
    Jx=psi_r[iy,ix]*dpi_dx-psi_i[iy,ix]*dpr_dx
    Jy=psi_r[iy,ix]*dpi_dy-psi_i[iy,ix]*dpr_dy
    return np.stack([Jx,Jy],axis=1)  # (N,2)

def vv_force_vec(vpos,vcharge,L):
    """Векторизованное вихрь-вихревое взаимодействие O(N²) numpy"""
    N=len(vpos)
    if N<2: return np.zeros((N,2))
    dr=vpos[:,np.newaxis,:]-vpos[np.newaxis,:,:]  # (N,N,2)
    dr=dr-L*np.round(dr/L)
    r=np.sqrt(dr[:,:,0]**2+dr[:,:,1]**2)  # (N,N)
    np.fill_diagonal(r,1e10)
    r=np.clip(r,0.5,None)
    qprod=vcharge[:,np.newaxis]*vcharge[np.newaxis,:]  # (N,N)
    mask=(r<vv_rng*2)
    w=np.where(mask, vv_str*qprod/r**2*np.exp(-r/vv_rng), 0)[:,:,np.newaxis]
    return np.sum(w*dr,axis=1)  # (N,2)

for step in range(steps):
    # TDGL
    A2=psi_r**2+psi_i**2
    if len(vpos)>0:
        cm=core_mask_vec(vpos,L,xi_core)
        pin_eff=pin_f+0.35*cm
    else:
        pin_eff=pin_f
    gl=-(alpha+beta*A2)
    lap_r=lap_open(psi_r); lap_i=lap_open(psi_i)
    dpr_dy=(np.roll(psi_r,-1,0)-np.roll(psi_r,1,0))/2
    dpi_dy=(np.roll(psi_i,-1,0)-np.roll(psi_i,1,0))/2
    cov_r=lap_r+2*H*xc*dpi_dy-(H*xc)**2*psi_r
    cov_i=lap_i-2*H*xc*dpr_dy-(H*xc)**2*psi_i
    F_r=gl*psi_r+gamma*cov_r-pin_eff*A2*psi_r+rng.normal(0,noise,(L,L))
    F_i=gl*psi_i+gamma*cov_i-pin_eff*A2*psi_i+rng.normal(0,noise,(L,L))
    denom=1.0+eta**2
    psi_r=np.clip(psi_r+dt*(F_r+eta*F_i)/denom,-4,4)
    psi_i=np.clip(psi_i+dt*(F_i-eta*F_r)/denom,-4,4)
    psi_r,psi_i=apply_bc(psi_r,psi_i,rng,noise)

    # Particles (векторизовано)
    if len(vpos)>0:
        J_loc=supercurrent_vec(psi_r,psi_i,vpos,L)
        J_tot=J_loc+J_tr[np.newaxis,:]
        # Magnus: v = q*(J × ẑ) = q*(-Jy, Jx)
        F_m=vcharge[:,np.newaxis]*B_z*np.stack([-J_tot[:,1],J_tot[:,0]],axis=1)
        F_vv=vv_force_vec(vpos,vcharge,L)
        F_n=rng.normal(0,noise_v,(len(vpos),2))
        vpos=(vpos+dt*(mob*(F_m+F_vv)+F_n))%L

    # Поглощение верхней границей
    alive=vpos[:,1]<L-2
    vpos=vpos[alive]; vcharge=vcharge[alive]

    # Нуклеация снизу
    while len(vpos)<N_pos+N_neg-6:
        q=rng.choice([-1.0,1.0])
        vpos=np.vstack([vpos,[rng.uniform(5,L-5),2.0]])
        vcharge=np.append(vcharge,q)

    A_map=np.sqrt(psi_r**2+psi_i**2)
    history['mean_A'].append(float(np.mean(A_map)))
    history['std_A'].append(float(np.std(A_map)))

    if step%500==0:
        pi=vcharge>0; ni=vcharge<0
        np_=np.sum(pi); nn=np.sum(ni)
        xp=float(np.mean(vpos[pi,0])) if np_>0 else L/2
        xn=float(np.mean(vpos[ni,0])) if nn>0 else L/2
        yp=float(np.mean(vpos[pi,1])) if np_>0 else L/2
        yn=float(np.mean(vpos[ni,1])) if nn>0 else L/2
        history['x_pos'].append(xp); history['x_neg'].append(xn)
        history['y_pos'].append(yp); history['y_neg'].append(yn)
        history['sep_x'].append(xp-xn); history['sep_y'].append(yp-yn)
        history['n_v'].append(len(vpos))
        if step in [0,3000,10000,29500]:
            snapshots[step]=(np.sqrt(psi_r**2+psi_i**2).copy(),
                             np.arctan2(psi_i,psi_r).copy(),
                             vpos.copy(),vcharge.copy())
        if step%5000==0:
            print(f"step {step:5d} | mean={np.mean(A_map):.4f} | std={np.std(A_map):.4f} | "
                  f"N={len(vpos)} (+{np_}/−{nn}) | Δx̄={xp-xn:.1f} Δȳ={yp-yn:.1f}")

fm=np.mean(history['mean_A'][-3000:])
fs=np.mean(history['std_A'][-3000:])
pi=vcharge>0; ni=vcharge<0
sep_x=history['sep_x'][-1]; sep_y=history['sep_y'][-1]
print(f"\nФинал: mean={fm:.4f} | std={fs:.4f} | N={len(vpos)}")
print(f"Δx̄={sep_x:.1f}px  {'✓✓' if abs(sep_x)>10 else '△'}")
print(f"Δȳ={sep_y:.1f}px  {'✓✓' if abs(sep_y)>10 else '△'}")

A_final=np.sqrt(psi_r**2+psi_i**2)
theta_final=np.arctan2(psi_i,psi_r)
dpr_dx=(np.roll(psi_r,-1,1)-np.roll(psi_r,1,1))/2
dpi_dx=(np.roll(psi_i,-1,1)-np.roll(psi_i,1,1))/2
dpr_dy2=(np.roll(psi_r,-1,0)-np.roll(psi_r,1,0))/2
dpi_dy2=(np.roll(psi_i,-1,0)-np.roll(psi_i,1,0))/2
Jx=psi_r*dpi_dx-psi_i*dpr_dx
Jy=(psi_r*dpi_dy2-psi_i*dpr_dy2)-H*xc*A_final**2
J_mag=np.sqrt(Jx**2+Jy**2)

sv=np.array(range(len(history['sep_x'])))*500
fig=plt.figure(figsize=(20,15))
gs=fig.add_gridspec(3,4,hspace=0.42,wspace=0.28)

ax=fig.add_subplot(gs[0,0])
ax.plot(history['mean_A'],'b-',lw=1.2,alpha=0.8)
ax.axhline(A_eq,color='r',ls='--',lw=1,label=f'теор={A_eq:.2f}')
ax.legend(fontsize=8); ax.set_title('mean |ψ|'); ax.grid(alpha=0.3)

ax=fig.add_subplot(gs[0,1])
ax.plot(history['std_A'],'g-',lw=1.2,alpha=0.8)
ax.set_title(f'std |ψ| = {fs:.4f}'); ax.grid(alpha=0.3)

ax=fig.add_subplot(gs[0,2])
ax.plot(sv,history['x_pos'],'b-',lw=2,label='x̄(+1)')
ax.plot(sv,history['x_neg'],'r-',lw=2,label='x̄(−1)')
ax.fill_between(sv,history['x_pos'],history['x_neg'],alpha=0.2,color='purple')
ax.axhline(L/2,color='gray',ls='--',alpha=0.4)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title(f'Magnus Δx̄={sep_x:.1f}px')

ax=fig.add_subplot(gs[0,3])
ax.plot(sv,history['sep_x'],'purple',lw=2.5,label='Δx̄(Magnus)')
ax.plot(sv,history['sep_y'],'teal',lw=2,ls='--',label='Δȳ(Hall)')
ax.axhline(0,color='gray',ls='--',lw=1)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title(f'Разделение\nΔx̄={sep_x:.1f}  Δȳ={sep_y:.1f}')

ax=fig.add_subplot(gs[1,0])
im=ax.imshow(A_final,origin='lower',cmap='hot',vmin=0)
p_pos=vpos[vcharge>0]; n_pos=vpos[vcharge<0]
if len(p_pos): ax.scatter(p_pos[:,0],p_pos[:,1],c='cyan',s=22,alpha=0.85,zorder=3)
if len(n_pos): ax.scatter(n_pos[:,0],n_pos[:,1],c='lime',s=22,alpha=0.85,zorder=3)
ax.set_title(r'$|\psi|$ + вихри'); fig.colorbar(im,ax=ax,fraction=0.046)

ax=fig.add_subplot(gs[1,1])
im=ax.imshow(theta_final,origin='lower',cmap='twilight',vmin=-np.pi,vmax=np.pi)
ax.set_title('Фаза θ'); fig.colorbar(im,ax=ax,fraction=0.046)

ax=fig.add_subplot(gs[1,2])
ax.imshow(J_mag,origin='lower',cmap='inferno',vmin=0)
if len(p_pos): ax.scatter(p_pos[:,0],p_pos[:,1],c='cyan',s=18,alpha=0.9,zorder=3)
if len(n_pos): ax.scatter(n_pos[:,0],n_pos[:,1],c='lime',s=18,alpha=0.9,zorder=3)
sq=8; yq,xq=np.arange(0,L,sq),np.arange(0,L,sq)
Xq,Yq=np.meshgrid(xq,yq)
mag=np.sqrt(Jx[::sq,::sq]**2+Jy[::sq,::sq]**2)+1e-9
ax.quiver(Xq,Yq,Jx[::sq,::sq]/mag,Jy[::sq,::sq]/mag,
          color='white',alpha=0.22,scale=32,width=0.002)
ax.set_title(f'|J| + вихри N={len(vpos)}')

ax=fig.add_subplot(gs[1,3])
ax.set_facecolor('#0a0a1a'); ax.set_xlim(0,L); ax.set_ylim(0,L)
if len(p_pos): ax.scatter(p_pos[:,0],p_pos[:,1],c='cyan',s=55,zorder=3,
                           label=f'+1  x̄={np.mean(p_pos[:,0]):.1f}')
if len(n_pos): ax.scatter(n_pos[:,0],n_pos[:,1],c='tomato',s=55,zorder=3,
                           label=f'−1  x̄={np.mean(n_pos[:,0]):.1f}')
if len(p_pos): ax.axvline(np.mean(p_pos[:,0]),color='cyan',ls='--',lw=2,alpha=0.7)
if len(n_pos): ax.axvline(np.mean(n_pos[:,0]),color='tomato',ls='--',lw=2,alpha=0.7)
ax.annotate('',xy=(L*0.85,8),xytext=(L*0.15,8),
            arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
ax.text(L/2,12,'J_transport →',color='white',ha='center',fontsize=9)
ax.legend(fontsize=8,loc='upper right')
ax.set_title(f'Финал t=30k: Magnus+Hall'); ax.set_aspect('equal')

snap_steps=sorted(snapshots.keys())
for k,sst in enumerate(snap_steps[:4]):
    ax=fig.add_subplot(gs[2,k])
    Asnap,_,vp_s,vc_s=snapshots[sst]
    ax.imshow(Asnap,origin='lower',cmap='hot',vmin=0,vmax=A_eq*1.05)
    pp=vp_s[vc_s>0]; np2=vp_s[vc_s<0]
    if len(pp): ax.scatter(pp[:,0],pp[:,1],c='cyan',s=12,alpha=0.9)
    if len(np2): ax.scatter(np2[:,0],np2[:,1],c='lime',s=12,alpha=0.9)
    if len(pp): ax.axvline(np.mean(pp[:,0]),color='cyan',ls='--',lw=1,alpha=0.6)
    if len(np2): ax.axvline(np.mean(np2[:,0]),color='lime',ls='--',lw=1,alpha=0.6)
    dx_s=(np.mean(pp[:,0])-np.mean(np2[:,0])) if len(pp)>0 and len(np2)>0 else 0
    ax.set_title(f't={sst}  Δx̄={dx_s:.1f}\n+{len(pp)}/−{len(np2)}',fontsize=8)

fig.suptitle(
    f'UAF+GL HYBRID 30k | TDGL+Particle-Tracker | J_tr={J_tr[0]}  H={H}  η={eta}\n'
    f'mean|ψ|={fm:.3f}  std={fs:.4f}  |  Δx̄={sep_x:.1f}px  Δȳ={sep_y:.1f}px',
    fontsize=12,fontweight='bold')
plt.savefig('/mnt/user-data/outputs/uaf_gl_hybrid_30k.png',dpi=250)
print("Saved.")
