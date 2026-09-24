import sys; import pathlib; _r=pathlib.Path(__file__).parents[1]; sys.path[:0]=[str(_r), str(_r/'tests')]
import numpy as np
from tlq.kernel import Derivation
from tlq import liquidation as LQ
from cases import liquidation
def run(policy, mu, xi_z, N=10, Q0=100.0, eta=1.0, phi=0.3):
    M=len(mu); xi=mu[:,None]+xi_z; X=50.0+np.c_[np.zeros(M),np.cumsum(xi,1)]
    Q=np.full(M,Q0); J=np.zeros(M); clipped=np.zeros(M,bool)
    for t in range(N):
        raw = Q.copy() if t==N-1 else policy(t,Q,X[:,t]-X[:,0])
        a=np.clip(raw,0,Q); clipped|=np.abs(a-raw)>1e-9
        J+=a*X[:,t]-eta*a**2; Q=Q-a
        if t<N-1: J-=phi*Q**2
    return J, clipped
d=Derivation(liquidation())
for r in (LQ.SHIFT,LQ.ADAPT,LQ.CONST_FORECAST,LQ.OPEN_LOOP,LQ.SCALE,LQ.SOLVE_RECURRENCE): d.apply(r)
u=d.solve().solution["u"]; fixed=lambda t,Q,disp: Q*u[t]
for s0 in (1.0, 3.0):
    P2=liquidation(context={"forecast":"bayes_gaussian","m0":0.0,"s0":s0,"sigma":1.0,"integrable":True})
    ctrl=Derivation(P2).apply(LQ.SHIFT).apply(LQ.ADAPT).apply(LQ.CE_MARTINGALE).solve().solution["controller"]
    learn=lambda t,Q,disp: np.array([ctrl(t,q,x) for q,x in zip(Q,disp)])
    rng=np.random.default_rng(1); M=20000; z=rng.normal(size=(M,10))
    for name,mu in ((f"mu~N(0,{s0:g}^2) (prior correct)", rng.normal(0,s0,M)), ("mu=0 (prior wrong)", np.zeros(M))):
        Ja,_=run(fixed,mu,z); Jb,cl=run(learn,mu,z); dlt=Jb-Ja
        print(f"prior sd {s0:g} | world {name:26s}: learning - declared = {dlt.mean():+7.2f} (paired se {dlt.std()/np.sqrt(M):.2f}); "
              f"ledger assumption 'no constraint binds' violated on {cl.mean():.1%} of paths")
