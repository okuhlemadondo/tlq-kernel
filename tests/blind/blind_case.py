"""Blind case, encoded strictly per PREREGISTRATION.md (I1-I8). Problem builder only."""
from tlq.finite import FiniteProblem, Site, Event, Read, ABSENT

V = {"L": 0.4, "H": 1.0}


def blind(variant="team", concept=None, c=0.05, clash_team=-0.2, clash_each=-0.1, distinct_labels=False):
    """distinct_labels=True is the POST-HOC correction of encoding error F14 (both types were labelled 't')."""
    la, lb = ("tA", "tB") if distinct_labels else ("t", "t")
    world = [(0.25, {"tA": a, "tB": b}) for a in ("L", "H") for b in ("L", "H")]                      # I1
    team = variant == "team"
    own = (lambda who: "team") if team else (lambda who: who)
    sig = ("none", "sigA", "sigB")
    sites = {
        "A1": Site("A1", own("A"), sig), "B1": Site("B1", own("B"), sig),                              # I2
        "SA": Site("SA", own("A"), ("sigA", "sigB"), program=lambda r: dict(r)["kind"]),
        "SB": Site("SB", own("B"), ("sigA", "sigB"), program=lambda r: dict(r)["kind"]),
        "A2": Site("A2", own("A"), (0, 1)), "B2": Site("B2", own("B"), (0, 1)),
    }
    seen = lambda v: "none" if v == ABSENT else v                                                    # I3
    events = [
        Event("eA1", "A1", (Read(la, "obs", lambda om, a: om["tA"]),)),
        Event("eB1", "B1", (Read(lb, "obs", lambda om, a: om["tB"]),)),
        Event("eSA", "SA", (Read("kind", "msg", lambda v: v, src="eA1"),), exists=lambda om, a: a.get("eA1") != "none"),
        Event("eSB", "SB", (Read("kind", "msg", lambda v: v, src="eB1"),), exists=lambda om, a: a.get("eB1") != "none"),
        Event("eA2", "A2", (Read(la, "obs", lambda om, a: om["tA"]), Read("other", "msg", seen, src="eSB"))),   # I4, I5
        Event("eB2", "B2", (Read(lb, "obs", lambda om, a: om["tB"]), Read("other", "msg", seen, src="eSA"))),
    ]

    def cost(a, who):
        return c * (a[f"e{who}1"] != "none")

    def team_pay(om, a):                                                                           # I6
        x, y = a["eA2"], a["eB2"]
        base = V[om["tA"]] if (x and not y) else V[om["tB"]] if (y and not x) else clash_team if (x and y) else 0.0
        return base - cost(a, "A") - cost(a, "B")

    def share(om, a, who):                                                                         # I7 (game)
        me, other = (a["eA2"], a["eB2"]) if who == "A" else (a["eB2"], a["eA2"])
        t = om["tA"] if who == "A" else om["tB"]
        return (V[t] if (me and not other) else clash_each if (me and other) else 0.0) - cost(a, who)

    if team:
        return FiniteProblem(world, sites, events, {"team": team_pay}, "team")
    return FiniteProblem(world, sites, events, {"A": lambda om, a: share(om, a, "A"), "B": lambda om, a: share(om, a, "B")},
                         concept or "pbe")                                                          # I8
