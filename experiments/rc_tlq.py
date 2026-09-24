"""RC / TLQ boundary experiments. Uses kernel v0.3 unchanged. Run: python3 experiments/rc_tlq.py [E1 E2 ...]"""
import sys, pathlib, math, time
import numpy as np
root = pathlib.Path(__file__).parents[1]
sys.path[:0] = [str(root), str(root / "tests")]
from tlq.finite import FiniteProblem, Site, Event, Read
from tlq.kernel import Derivation, Rejected
from tlq.rules_finite import TEAM_ENUM, BEHAVIOURAL_OPT, NORMAL_FORM, IESDS, PURE_NASH
import tlq.search as SE
from tlq import liquidation as LQ
from cases import parent_child, driver, liquidation


def team_value(P):
    r = Derivation(P).apply(TEAM_ENUM()).solve()
    return r.solutions[0]["value"], r.solutions[0]["policy"]


# ------------------------------------------------------------------ E1: observable composition as a policy inside a problem
def probes(target="xor", cost=0.1):
    """Choose which probes to run (none / X / Y / XY), then guess the target. Probe results are WORLD reads that
    exist only if the probe action was taken: composing observables is an action, its value is set by the task."""
    world = [(0.25, {"x": x, "y": y}) for x in (0, 1) for y in (0, 1)]
    if target == "copy":                       # y is a copy of x: the probes are substitutes
        world = [(0.5, {"x": x, "y": x}) for x in (0, 1)]
    tgt = (lambda om: om["x"] ^ om["y"]) if target == "xor" else (lambda om: om["x"])
    sites = {"probe": Site("probe", "team", ("none", "X", "Y", "XY")), "guess": Site("guess", "team", (0, 1))}
    events = [Event("eP", "probe", ()),
              Event("eG", "guess", (Read("ox", "obs", lambda om, a: om["x"] if "X" in a["eP"] else "-"),
                                    Read("oy", "obs", lambda om, a: om["y"] if "Y" in a["eP"] else "-")))]
    pay = {"team": lambda om, a: (1.0 if a["eG"] == tgt(om) else 0.0) - cost * (a["eP"] != "none") * len(a["eP"].replace("none", ""))}
    return FiniteProblem(world, sites, events, pay, "team")


def E1():
    print("E1  observable composition is an action inside the problem; the task decides its value")
    for tgt in ("xor", "copy"):
        P = probes(tgt)
        v, pol = team_value(P)
        rep = SE.search(P, diagnose=False)
        print(f"    target={tgt:4s}: optimal value {v:.2f}, probes chosen: {list(pol['probe'].values())[0]:4s} | search status {rep.status}, "
              f"derivations {[[r.name for r in f.rules] for f in rep.found][:3]}")


# ------------------------------------------------------------------ E2: adding access in a single-owner team never lowers value
def E2():
    print("E2  access-adding arrows in single-owner teams: value is monotone (policy-class inclusion)")
    a, _ = team_value(parent_child()); b, _ = team_value(parent_child(parent_reads_child_record=True))
    print(f"    parent/child: without child's record {a:.4f} -> with it {b:.4f}")
    r1 = Derivation(driver(memory=False, coin="fresh")).apply(BEHAVIOURAL_OPT()).solve().solutions[0]["value"]
    r2 = Derivation(driver(memory=True, coin="fresh")).apply(TEAM_ENUM()).solve().solutions[0]["value"]
    print(f"    driver: forgetful {r1:.4f} -> with a visit counter {r2:.4f}")
    # but derivability is NOT monotone in access: adding a world-mediated read blocks COMMON_INFO
    from cases import channel
    for label, P in (("record route", channel("record")), ("world route", channel("world"))):
        ok = True
        try:
            from tlq.rules_finite import COMMON_INFO
            Derivation(P).apply(COMMON_INFO())
        except Rejected as e:
            ok = False
        print(f"    COMMON_INFO certifiable on the {label}: {ok}")


# ------------------------------------------------------------------ E3: information hurts in games (risk sharing)
def insurance(informed):
    world = [(0.5, {"s": s}) for s in (0, 1)]
    sites = {"P1": Site("P1", "one", (0, 1)), "P2": Site("P2", "two", (0, 1))}
    reads = (Read("s", "obs", lambda om, a: om["s"]),) if informed else ()
    events = [Event("e1", "P1", reads), Event("e2", "P2", reads)]

    def u(om, a, who):
        if a["e1"] == 1 and a["e2"] == 1:
            return math.sqrt(0.5)                    # both agree to share: each gets half for sure
        mine = (om["s"] == 0) if who == 1 else (om["s"] == 1)
        return math.sqrt(1.0 if mine else 0.0)
    return FiniteProblem(world, sites, events, {"one": lambda om, a: u(om, a, 1), "two": lambda om, a: u(om, a, 2)}, "nash")


def E3():
    print("E3  more information can lower EVERY player's equilibrium value (Hirshleifer effect)")
    for informed in (False, True):
        r = Derivation(insurance(informed)).apply(NORMAL_FORM()).apply(PURE_NASH()).solve()
        vals = sorted({(round(s["vA"], 4), round(s["vB"], 4)) for s in r.solutions}, reverse=True)
        print(f"    state observed before choosing: {informed!s:5s} -> pure-equilibrium payoffs {vals}")


# ------------------------------------------------------------------ E4: where the uncertainty is placed (declared vs internalized)
def simulate(policy, mu_draw, M=200000, N=10, Q0=100.0, eta=1.0, phi=0.3, seed=0):
    rng = np.random.default_rng(seed)
    mu = mu_draw(rng, M)
    xi = rng.normal(mu[:, None], 1.0, size=(M, N))
    X = 50.0 + np.c_[np.zeros(M), np.cumsum(xi, 1)]
    Q = np.full(M, Q0); J = np.zeros(M)
    for t in range(N):
        a = Q.copy() if t == N - 1 else np.clip(policy(t, Q, X[:, t] - X[:, 0]), 0, Q)
        J += a * X[:, t] - eta * a ** 2; Q = Q - a
        if t < N - 1:
            J -= phi * Q ** 2
    return J.mean(), J.std() / np.sqrt(M)


def E4():
    print("E4  the same world, uncertainty declared (ledger) vs internalized (prior): 'learning' as inner belief update")
    P1 = liquidation()
    d1 = Derivation(P1)
    for r in (LQ.SHIFT, LQ.ADAPT, LQ.CONST_FORECAST, LQ.OPEN_LOOP, LQ.SCALE, LQ.SOLVE_RECURRENCE):
        d1.apply(r)
    r1 = d1.solve()
    u = r1.solution["u"]
    fixed = lambda t, Q, disp: Q * u[t]
    P2 = liquidation(context={"forecast": "bayes_gaussian", "m0": 0.0, "s0": 1.0, "sigma": 1.0, "integrable": True})
    r2 = Derivation(P2).apply(LQ.SHIFT).apply(LQ.ADAPT).apply(LQ.CE_MARTINGALE).solve()
    ctrl = r2.solution["controller"]
    ce = lambda t, Q, disp: np.array([ctrl(t, q, d) for q, d in zip(Q, disp)]) if np.ndim(Q) else ctrl(t, Q, disp)
    print(f"    ledger of the declared formulation: {[o for _, o in r1.ledger]}")
    print(f"    ledger of the internalized formulation: {[o for _, o in r2.ledger]}")
    for name, draw in (("drift really random, mu~N(0,1)", lambda g, M: g.normal(0, 1, M)), ("drift really 0", lambda g, M: np.zeros(M))):
        a, sa = simulate(fixed, draw, M=20000)
        b, sb = simulate(lambda t, Q, d: np.array([ctrl(t, q, x) for q, x in zip(Q, d)]), draw, M=20000)
        print(f"    {name:34s}: declared-constant policy {a:9.2f} (se {sa:.2f})  learning controller {b:9.2f} (se {sb:.2f})  gap {b - a:+.2f}")


# ------------------------------------------------------------------ E5: a derivation is a reusable program whose domain is its obligations
def replay(P, rules):
    d = Derivation(P)
    try:
        for r in rules:
            d.apply(r)
        res = d.solve()
        return "ok", res
    except Rejected as e:
        return sorted(e.names())[0], None


def E5(n=200):
    print("E5  derivation transfer: replaying a found derivation on new instances (generalization domain = obligations)")
    rng = np.random.default_rng(3)
    D = [LQ.SHIFT, LQ.ADAPT, LQ.CONST_FORECAST, LQ.OPEN_LOOP, LQ.SCALE, LQ.SOLVE_RECURRENCE]
    D2 = [LQ.SHIFT, LQ.ADAPT, LQ.CONST_FORECAST, LQ.OPEN_LOOP, LQ.SOLVE_RECURRENCE]
    from scipy.optimize import minimize
    outcomes, outcomes2, errs = {}, {}, []
    for _ in range(n):
        mu = 0.0 if rng.random() < 0.5 else float(rng.normal(0, 30))
        P = LQ.Liquidation(N=int(rng.integers(3, 20)), Q0=float(rng.uniform(10, 200)), eta=float(rng.uniform(0.2, 3)),
                           phi=float(rng.uniform(0.01, 1)), context={"forecast": "constant", "mu": mu, "integrable": True})
        k, res = replay(P, D); outcomes[k] = outcomes.get(k, 0) + 1
        k2, res2 = replay(P, D2); outcomes2[k2] = outcomes2.get(k2, 0) + 1
        for rr in (res, res2):
            if rr is not None:
                f = lambda q: -LQ.objective(P, np.r_[P.Q0, q, 0.0], mu)
                x = minimize(f, np.linspace(P.Q0, 0, P.N + 1)[1:-1], method="BFGS", options={"gtol": 1e-9}).x
                errs.append(np.abs(rr.solution["Q"] - np.r_[P.Q0, x, 0.0]).max() / P.Q0)
    print(f"    derivation with SCALE on {n} random instances: {outcomes}")
    print(f"    derivation without SCALE:                     {outcomes2}")
    print(f"    every accepted replay matches an independent optimizer: max relative path error {max(errs):.1e}")
    ok = fail = 0
    for _ in range(50):
        q, c = float(rng.uniform(0.55, 0.95)), float(rng.uniform(0.0, 0.45))
        k, res = replay(parent_child(cost=c, q=q), [__import__("tlq.rules_finite", fromlist=["x"]).COMMON_INFO(),
                                                    __import__("tlq.rules_finite", fromlist=["x"]).COORDINATOR_DP()])
        v, _ = team_value(parent_child(cost=c, q=q))
        ok += (k == "ok" and abs(res.solution["value"] - v) < 1e-12)
    k, _ = replay(parent_child(auditor=True), [__import__("tlq.rules_finite", fromlist=["x"]).COMMON_INFO()])
    print(f"    common-information derivation on 50 random parent/child instances: {ok}/50 valid and equal to brute force; "
          f"on a structurally different instance: rejected at [{k}]")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["E1", "E2", "E3", "E4", "E5"]):
        t = time.time(); globals()[name](); print(f"    ({time.time() - t:.1f}s)\n")
