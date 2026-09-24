"""Problem builders for the forcing cases. Problems only — no solutions are encoded here."""
from tlq.finite import FiniteProblem, Site, Event, Read, ABSENT
from tlq.liquidation import Liquidation


# ---------------------------------------------------------------- C1 / C2 liquidation
def liquidation(**kw):
    base = dict(N=10, Q0=100.0, eta=1.0, phi=0.3)
    base.update(kw)
    return Liquidation(**base)


# ---------------------------------------------------------------- C3 parent/child team (13/20)
def parent_child(cost=0.2, q=0.75, parent_reads_child_record=False, auditor=False):
    world = [(0.5 * (q if y == b else 1 - q), {"b": b, "y": y}) for b in (0, 1) for y in (0, 1)]
    sites = {"child": Site("child", "team", (0, 1)), "parent": Site("parent", "team", (0, 1))}
    reads_parent = [Read("c", "msg", lambda v: v, src="e_child")]
    if parent_reads_child_record:
        reads_parent.append(Read("y_fwd", "msg", lambda r: r, src="e_child", of="record"))
    events = [Event("e_child", "child", (Read("y", "obs", lambda om, a: om["y"]),)),
              Event("e_parent", "parent", tuple(reads_parent))]
    if auditor:   # a later site that does NOT read the child's action: the action is no longer common information
        sites["auditor"] = Site("auditor", "team", (0, 1))
        events.append(Event("e_aud", "auditor", ()))
    pay = {"team": lambda om, a: (1.0 if a["e_parent"] == om["b"] else 0.0) - cost * a["e_child"]}
    return FiniteProblem(world, sites, events, pay, "team")


# ---------------------------------------------------------------- C4 absent-minded driver
def driver(memory=False, coin="fresh", two_sites=False):
    acts = ("exit", "continue")
    if two_sites:   # the conflation: each evaluation modelled as its own site
        sites = {"d1": Site("d1", "me", acts, memory=memory, coin=coin), "d2": Site("d2", "me", acts, memory=memory, coin=coin)}
        events = [Event("x1", "d1"), Event("x2", "d2", exists=lambda om, a: a.get("x1") == "continue")]
    else:
        sites = {"driver": Site("driver", "me", acts, memory=memory, coin=coin)}
        events = [Event("x1", "driver"), Event("x2", "driver", exists=lambda om, a: a.get("x1") == "continue")]

    def pay(om, a):
        if a["x1"] == "exit":
            return 0.0
        return 4.0 if a["x2"] == "exit" else 1.0
    return FiniteProblem([(1.0, {})], sites, events, {"me": pay}, "team")


# ---------------------------------------------------------------- C5 silence vs absence
def silence(cost=0.1, q=0.8, channel=True, silence_allowed=True):
    world = [(0.5 * (q if y == b else 1 - q), {"b": b, "y": y}) for b in (0, 1) for y in (0, 1)]
    child_acts = ("S", "0", "1") if silence_allowed else ("0", "1")
    sites = {"child": Site("child", "team", child_acts), "parent": Site("parent", "team", (0, 1))}
    reads = (Read("m", "msg", lambda v: v, src="e_child"),) if channel else ()
    events = [Event("e_child", "child", (Read("y", "obs", lambda om, a: om["y"]),)), Event("e_parent", "parent", reads)]
    pay = {"team": lambda om, a: (1.0 if a["e_parent"] == om["b"] else 0.0) - cost * (a["e_child"] != "S")}
    return FiniteProblem(world, sites, events, pay, "team", relevant={"parent": lambda om: om["b"]})


# ---------------------------------------------------------------- C6 endogenous absence (runtime-created locus)
def spawning(prior=0.5, endogenous=True):
    world = [((prior if x else 1 - prior) * 0.25, {"x": x, "w": w, "z": z}) for x in (0, 1) for w in (0, 1) for z in (0, 1)]
    trigger = "x" if endogenous else "z"
    sites = {
        "P": Site("P", "team", ("spawn", "no"), program=lambda r: "spawn" if dict(r)[trigger] == 1 else "no"),
        "C": Site("C", "team", ("ping", "silent"), program=lambda r: "ping" if dict(r)["w"] == 1 else "silent"),
        "R": Site("R", "team", (0, 1)),
    }
    events = [
        Event("eP", "P", (Read(trigger, "obs", lambda om, a: om[trigger]),)),
        Event("eC", "C", (Read("w", "obs", lambda om, a: om["w"]),), exists=lambda om, a: a.get("eP") == "spawn"),
        Event("eR", "R", (Read("c", "msg", lambda v: "ping" if v == "ping" else "none", src="eC"),)),
    ]
    pay = {"team": lambda om, a: 1.0 if a["eR"] == om["x"] else 0.0}
    return FiniteProblem(world, sites, events, pay, "team", relevant={"R": lambda om: om["x"]})


# ---------------------------------------------------------------- C7 non-confluence
def weak_dominance_game():
    import numpy as np
    from tlq.rules_finite import Bimatrix
    U = {("T", "L"): (1, 1), ("T", "R"): (0, 0), ("M", "L"): (1, 1), ("M", "R"): (2, 1), ("B", "L"): (0, 0), ("B", "R"): (2, 1)}
    rows, cols = ["T", "M", "B"], ["L", "R"]
    A = np.array([[U[(r, c)][0] for c in cols] for r in rows], float)
    B = np.array([[U[(r, c)][1] for c in cols] for r in rows], float)
    return Bimatrix(("row", "col"), rows, cols, A, B)


def strict_dominance_game():
    import numpy as np
    from tlq.rules_finite import Bimatrix
    rows, cols = ["a", "b", "c"], ["x", "y", "z"]
    A = np.array([[3, 1, 2], [2, 0, 1], [1, 2, 0]], float)   # b strictly dominated by a
    B = np.array([[1, 3, 0], [2, 1, 0], [0, 2, 1]], float)   # z strictly dominated by y... after b removed
    return Bimatrix(("row", "col"), rows, cols, A, B)


# ---------------------------------------------------------------- C8 composite (round 6)
def composite(cd=0.2, eps=0.1, leak=None, g=1.0, lam=1.0, ell=1.0, alpha=0.5, k=0.3, psi=0.6, m=0.1):
    """leak: None | 'world' (dark order visible as a noisy dip: round 6's actual model) | 'arrow' (record arrow D -> B)."""
    world = []
    for th in ("H", "N"):
        for fill in (True, False):
            for n1 in (0, 1):
                for n2 in (0, 1):
                    p = 0.5 * (psi if fill else 1 - psi) * (eps if n1 else 1 - eps) * (eps if n2 else 1 - eps)
                    if p > 0:
                        world.append((p, {"theta": th, "fill": fill, "n1": n1, "n2": n2}))

    def dip1(om, a):
        return int(a.get("eA") == "lit" or (leak == "world" and a.get("eA") == "dark" and om["theta"] == "H")) ^ om["n1"]

    def dip2(om, a):
        return int(a.get("eA2") == "sell") ^ om["n2"]

    sites = {
        "A": Site("A", "A", ("lit", "dark", "none")),
        "C": Site("C", "A", ("sell",)),
        "D": Site("D", "A", ("filled", "unfilled"), program=lambda r: "filled" if dict(r)["fill"] else "unfilled"),
        "A2": Site("A2", "A", ("sell", "hold"), program=lambda r: "sell" if dict(r)["report"] == "unfilled" else "hold"),
        "B1": Site("B1", "B", (0, 1)),
        "B2": Site("B2", "B", (0, 1)),
    }
    arrow = (Read("dark", "msg", lambda v: "none" if v == ABSENT else "order", src="eD"),) if leak == "arrow" else ()
    events = [
        Event("eA", "A", (Read("theta", "obs", lambda om, a: om["theta"]),)),
        Event("eC", "C", (), exists=lambda om, a: a.get("eA") == "lit"),
        # a dark broker is only spawned when there is a block to work (type H); for N, 'dark' just pays the fee
        Event("eD", "D", (Read("fill", "obs", lambda om, a: om["fill"]),), exists=lambda om, a: a.get("eA") == "dark" and om["theta"] == "H"),
        Event("eB1", "B1", (Read("o1", "obs", dip1),) + arrow),
        Event("eA2", "A2", (Read("report", "msg", lambda v: v, src="eD"),), exists=lambda om, a: "eD" in a),
        Event("eB2", "B2", (Read("o1", "obs", dip1), Read("o2", "obs", dip2)) + arrow),
    ]

    def parts(om, a):
        real1 = om["theta"] == "H" and (a["eA"] == "lit" or (leak is not None and a["eA"] == "dark"))
        real2 = a.get("eA2") == "sell"
        act = a["eA"]
        if act == "lit":
            cost = 0.0 if om["theta"] == "H" else cd
        elif act == "dark":
            cost = k + (m if real2 else 0.0)
        else:
            cost = 0.0 if om["theta"] == "N" else 10.0
        return real1, real2, cost

    def uA(om, a):
        r1, r2, cost = parts(om, a)
        return -cost + (-lam if r1 else alpha * ell) * a["eB1"] + (-lam if r2 else alpha * ell) * a["eB2"]

    def uB(om, a):
        r1, r2, _ = parts(om, a)
        return (g if r1 else -ell) * a["eB1"] + (g if r2 else -ell) * a["eB2"]

    return FiniteProblem(world, sites, events, {"A": uA, "B": uB}, "pbe")


# ---------------------------------------------------------------- C9 world route vs record route
def channel(route="world", e=0.0, gamma=0.0, bystander=False, q=0.8, c=0.1):
    world = [(0.5 * (q if y == b else 1 - q) * (e if n else 1 - e), {"b": b, "y": y, "n": n})
             for b in (0, 1) for y in (0, 1) for n in (0, 1) if (e if n else 1 - e) > 0]
    sites = {"C": Site("C", "team", (0, 1)), "P": Site("P", "team", (0, 1))}
    if route == "world":
        pr = Read("o", "obs", lambda om, a: a["eC"] ^ om["n"])
    else:
        pr = Read("o", "msg", lambda v: v, src="eC")
    events = [Event("eC", "C", (Read("y", "obs", lambda om, a: om["y"]),)), Event("eP", "P", (pr,))]
    if bystander:
        sites["W"] = Site("W", "team", (0, 1))
        events.append(Event("eW", "W", (Read("o", "obs", lambda om, a: a["eC"] ^ om["n"]),)))
    g = gamma if route == "world" else 0.0      # a record transfer moves no prices: only the world route has side effects
    pay = {"team": lambda om, a: (1.0 if a["eP"] == om["b"] else 0.0) - c * a["eC"] + g * a["eC"] * (1 - 2 * a["eP"])}
    return FiniteProblem(world, sites, events, pay, "team")


def route_choice(single_record=False, q=0.8):
    """Sender chooses a route: 'rec' (record message), 'push' (world signal, cheaper), or 'none'."""
    world = [(0.5 * (q if y == b else 1 - q), {"b": b, "y": y}) for b in (0, 1) for y in (0, 1)]
    sites = {"C": Site("C", "team", ("none", "rec", "push")), "P": Site("P", "team", (0, 1))}
    c_reads = () if single_record else (Read("y", "obs", lambda om, a: om["y"]),)
    events = [Event("eC", "C", c_reads),
              Event("eP", "P", (Read("m", "msg", lambda v: int(v == "rec"), src="eC"), Read("o", "obs", lambda om, a: int(a["eC"] == "push"))))]
    cost = {"none": 0.0, "rec": 0.12, "push": 0.05}
    pay = {"team": lambda om, a: (1.0 if a["eP"] == om["b"] else 0.0) - cost[a["eC"]]}
    return FiniteProblem(world, sites, events, pay, "team")


# ---------------------------------------------------------------- execution semantics: a cycle
def cycle():
    sites = {"s1": Site("s1", "team", (0, 1)), "s2": Site("s2", "team", (0, 1))}
    events = [Event("e1", "s1", (Read("a2", "msg", lambda v: v, src="e2"),)), Event("e2", "s2", (Read("a1", "msg", lambda v: v, src="e1"),))]
    return FiniteProblem([(1.0, {})], sites, events, {"team": lambda om, a: 0.0}, "team")
