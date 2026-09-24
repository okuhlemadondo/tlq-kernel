"""Certified transformations for finite problems. Each rule is general over its problem class."""
from __future__ import annotations
from dataclasses import dataclass, replace
import itertools
import numpy as np
from .kernel import Transformation, Obligation, Explicit, S, L, D, V
from .finite import FiniteProblem, Read, ABSENT

TOL = 1e-9


def ok(cond, yes, no):
    return (True, yes) if cond else (False, no)


def is_finite(P):
    return ok(getattr(P, "kind", None) == "finite", "finite problem", "not a finite problem")


def team_ok(P):
    return ok(P.concept == "team" and len(P.payoffs) == 1, "single-owner team", f"concept={P.concept}, owners={list(P.payoffs)}")


def is_finite_team(P):
    """Applicability includes the solution concept: a team solver is not ABOUT a game (wrong kind, not a failed certificate)."""
    if getattr(P, "kind", None) != "finite":
        return False, "not a finite problem"
    return team_ok(P)


# =====================================================================================  team solvers
def TEAM_ENUM():
    """Terminal: best pure profile by enumeration. Certified only when pure plans are value-complete."""
    def obligations(P):
        obs = [Obligation("execution semantics well-founded", S, lambda P: P.well_founded()),
               Obligation("single-owner team", S, team_ok)]
        if any(s.coin == "fresh" for s in P.decision_sites()):
            obs.append(Obligation("pure plans are value-complete despite fresh randomness", L,
                                  lambda P: ok(not P.repeats_site_record(),
                                               "no (site, record) is evaluated twice on any path: value is multilinear, a vertex is optimal",
                                               "a (site, record) is evaluated twice on a path: value is not multilinear and randomization can strictly help"),
                                  lemma="multilinear function on a product of simplices attains its max at a vertex"))
        if any(s.coin == "persistent" for s in P.decision_sites()):
            obs.append(Obligation("mixtures of pure plans cannot beat the best pure plan", L,
                                  lambda P: (True, "value is linear in the mixture"), lemma="linearity of expectation"))
        return obs

    def forward(P):
        owner = next(iter(P.payoffs))
        best, sols = None, []
        for pol in P.pure_policies():
            v = P.value(pol)[owner]
            if best is None or v > best + TOL:
                best, sols = v, [pol]
            elif abs(v - best) <= TOL:
                sols.append(pol)
        return Explicit([{"policy": s, "value": best} for s in sols], note=f"{len(sols)} optimal pure profiles")

    def cost(P):
        n = 1
        for s, recs in P.reachable_records().items():
            if P.sites[s].program is None and len(P.sites[s].actions) > 1:
                n *= len(P.sites[s].actions) ** len(recs)
        return n

    return Transformation("TEAM_ENUM", frozenset({"sound", "complete"}), is_finite_team, obligations, forward, terminal=True, cost=cost)


def BEHAVIOURAL_OPT(starts=24, seed=0):
    """Terminal: optimize behavioural probabilities per (site, record). Admissible only with fresh randomness."""
    from scipy.optimize import minimize

    def obligations(P):
        return [Obligation("single-owner team", S, team_ok),
                Obligation("behavioural strategies admissible (fresh randomness at every decision site)", S,
                           lambda P: ok(all(s.coin == "fresh" for s in P.decision_sites()), "every decision site has a fresh coin",
                                        "some decision site has no fresh randomness")),
                Obligation("global optimum (multistart agreement)", V,
                           lambda P, s: ok(s is not None and s["agree"] >= 3, f"{s and s['agree']} starts agree",
                                           "multistart did not agree: optimum not certified"), required=False, for_property="sound"),
                Obligation("global optimum (multistart agreement) ", V,
                           lambda P, s: ok(s is not None and s["agree"] >= 3, "", "not certified"), required=False, for_property="complete")]

    def forward(P):
        owner = next(iter(P.payoffs))
        recs = P.reachable_records()
        slots = [(s.name, r) for s in P.decision_sites() for r in recs[s.name]]
        acts = {s.name: s.actions for s in P.decision_sites()}
        sizes = [len(acts[sn]) for sn, _ in slots]

        def unpack(x):
            pol, k = {sn: {} for sn in acts}, 0
            for (sn, r), n in zip(slots, sizes):
                z = np.exp(x[k:k + n] - x[k:k + n].max()); z /= z.sum()
                pol[sn][r] = dict(zip(acts[sn], z)); k += n
            return pol
        f = lambda x: -P.value(unpack(x))[owner]
        rng = np.random.default_rng(seed)
        runs = [minimize(f, rng.normal(0, 2, sum(sizes)), method="Nelder-Mead",
                         options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}) for _ in range(starts)]
        best = min(runs, key=lambda r: r.fun)
        agree = sum(abs(r.fun - best.fun) < 1e-7 for r in runs)
        return Explicit([{"policy": unpack(best.x), "value": -best.fun, "agree": agree}])

    return Transformation("BEHAVIOURAL_OPT", frozenset({"sound", "complete"}), is_finite_team, obligations, forward, terminal=True)


# =====================================================================================  common information
@dataclass
class CoordinatorProblem:
    base: FiniteProblem
    public: dict            # event -> bool (its action is read, injectively, by every later event)
    kind: str = "coordinator"


def _injective_on_actions(P, read):
    src_site = P.sites[next(e.site for e in P.events if e.name == read.src)]
    vals = [read.fn(a) for a in src_site.actions]
    return len(set(map(repr, vals))) == len(vals)


def _public_events(P):
    pub = {}
    for i, e in enumerate(P.events):
        later = P.events[i + 1:]
        pub[e.name] = bool(later) and all(
            any(r.kind == "msg" and r.src == e.name and r.of == "action" and _injective_on_actions(P, r) for r in l.reads)
            for l in later)
    return pub


def COMMON_INFO():
    """Many decision sites -> one virtual coordinator choosing prescriptions from common information."""
    def structure(P):
        if any(e.exists is not None for e in P.events):
            return False, "generated (conditional) events: this implementation needs a fixed event list"
        per_site = {}
        for e in P.events:
            per_site[e.site] = per_site.get(e.site, 0) + 1
        if any(n > 1 for n in per_site.values()) or any(s.memory for s in P.sites.values()):
            return False, "each decision site must be evaluated once and hold no memory in this implementation"
        return True, "fixed events, one evaluation per site"

    def partial_history_sharing(P):
        pub = _public_events(P)
        obs_readers = {}
        for e in P.events:
            for r in e.reads:
                if r.kind == "obs":
                    obs_readers.setdefault(r.label, set()).add(e.name)
        for e in P.events:
            for r in e.reads:
                if r.kind == "msg" and not (r.of == "action" and pub.get(r.src)):
                    return False, f"{e.name} reads {r.src}'s {r.of} without it being common to all later sites"
                if r.kind == "obs" and len(obs_readers[r.label]) > 1:
                    return False, f"world item {r.label} is read by several sites without being common"
        return True, "every read is either common (public action) or private to its site"

    def no_world_mediated_flow(P):
        for e in P.events:
            for r in e.reads:
                if r.kind != "obs":
                    continue
                for _, om in P.world:
                    vals = {repr(r.fn(om, acts)) for acts in _all_action_paths(P, om)}
                    if len(vals) > 1:
                        return False, (f"world item {r.label} read at {e.name} depends on other decisions: "
                                       "information flows through the world, outside the common/private split")
        return True, "every private observation is a function of the world outcome alone"

    def obligations(P):
        return [Obligation("single-owner team", S, team_ok),
                Obligation("execution semantics well-founded", S, lambda P: P.well_founded()),
                Obligation("fixed event structure", S, structure),
                Obligation("no world-mediated propagation between sites", S, no_world_mediated_flow),
                Obligation("partial history sharing", S, partial_history_sharing)]

    def forward(P):
        return CoordinatorProblem(P, _public_events(P))

    def lift(P, sol):
        return sol   # the coordinator DP already lifts prescriptions to site policies (see COORDINATOR_DP)

    return Transformation("COMMON_INFO", frozenset({"sound", "complete"}), is_finite_team, obligations, forward, lift)


def COORDINATOR_DP():
    """Terminal: dynamic programming over common information with prescriptions as actions."""
    def applies(C):
        return ok(getattr(C, "kind", None) == "coordinator", "coordinator problem", "not a coordinator problem")

    def forward(C):
        P = C.base
        owner = next(iter(P.payoffs))
        pay = P.payoffs[owner]

        def private(ev, omega):
            return tuple((r.label, r.fn(omega, {})) for r in ev.reads if r.kind == "obs")

        def rec(k, belief):
            if k == len(P.events):
                return sum(p * pay(om, acts) for p, om, acts in belief), {}
            ev = P.events[k]
            site = P.sites[ev.site]
            privs = sorted({private(ev, om) for _, om, _ in belief}, key=repr)
            best = None
            actions = site.actions if (site.program is None and len(site.actions) > 1) else None
            presc_space = (dict(zip(privs, c)) for c in itertools.product(actions, repeat=len(privs))) if actions else [None]
            for g in presc_space:
                nxt = []
                for p, om, acts in belief:
                    a = g[private(ev, om)] if g is not None else site.program(P.record(ev, om, acts, {}, {})) if site.program else site.actions[0]
                    nxt.append((p, om, {**acts, ev.name: a}))
                if C.public[ev.name]:
                    groups = {}
                    for item in nxt:
                        groups.setdefault(item[2][ev.name], []).append(item)
                    val, sub = 0.0, {}
                    for a, grp in groups.items():
                        v, strat = rec(k + 1, grp); val += v; sub[a] = strat
                else:
                    val, strat = rec(k + 1, nxt); sub = {None: strat}
                if best is None or val > best[0] + TOL:
                    best = (val, {"presc": g, "next": sub})
            return best

        val, tree = rec(0, [(p, om, {}) for p, om in P.world])
        # lift prescriptions to site policies: walk the tree along public actions
        policy = {s.name: {} for s in P.decision_sites()}

        def walk(k, node, pub_hist):
            if k == len(P.events):
                return
            ev = P.events[k]
            g = node["presc"]
            if g is not None:
                for priv, a in g.items():
                    # the site's record = public msgs (from pub_hist) + private obs, in read order
                    rec_items = []
                    for r in ev.reads:
                        if r.kind == "msg":
                            rec_items.append((r.label, r.fn(pub_hist[r.src])))
                        else:
                            rec_items.append((r.label, dict(priv)[r.label]))
                    policy[ev.site][tuple(rec_items)] = a
            for a, child in node["next"].items():
                walk(k + 1, child, {**pub_hist, ev.name: a} if a is not None else pub_hist)
        walk(0, tree, {})
        return Explicit([{"policy": policy, "value": val}])

    return Transformation("COORDINATOR_DP", frozenset({"sound", "complete"}), applies, lambda C: [], forward, terminal=True)


# =====================================================================================  absence / silence
def NULL_AS_ABSENCE(site, label, null, uninformed_action):
    """Claim: a null reading on `label` carries no information for `site`, so it may act as if uninformed."""
    def applies(P):
        return ok(getattr(P, "kind", None) == "finite" and any(e.site == site and any(r.label == label for r in e.reads) for e in P.events),
                  "site reads the item", "site does not read this item")

    def uninformative(P):
        rel = P.relevant[site]
        others = [s.name for s in P.decision_sites() if s.name != site]
        for pol in P.pure_policies(others) if others else [{}]:
            pol = {**pol, site: "uniform"}
            joint = {}
            for pw, om in P.world:
                for p, acts, recs, _ in P.paths(pol, om):
                    for e in P.events:
                        if e.site == site and e.name in recs:
                            isnull = dict(recs[e.name]).get(label) == null
                            joint[(rel(om), isnull)] = joint.get((rel(om), isnull), 0) + pw * p
            states = {k[0] for k in joint}
            pn = sum(v for k, v in joint.items() if k[1])
            if pn <= 0:
                continue
            tot = sum(joint.values())
            for x in states:
                prior = sum(v for k, v in joint.items() if k[0] == x) / tot
                post = joint.get((x, True), 0) / pn
                if abs(prior - post) > 1e-9:
                    return False, (f"under another site's admissible program {({k: v for k, v in pol.items() if k != site} or 'fixed programs')}, "
                                   f"P({x} | {label}={null}) = {post:.4f} != prior {prior:.4f}: the null value is informative")
        return True, "the null value leaves the task-relevant state at its prior under every admissible program"

    def obligations(P):
        return [Obligation("null reading is uninformative under every admissible program", L, uninformative,
                           lemma="Bayes: a signal independent of the state leaves the posterior at the prior")]

    def forward(P):
        pred = lambda rec, _l=label, _n=null: dict(rec).get(_l) == _n
        return P.copy(restrictions=P.restrictions + [(site, pred, uninformed_action)])

    def lift(P, sol):
        if "policy" not in sol:
            return sol
        pol = {k: (dict(v) if isinstance(v, dict) else v) for k, v in sol["policy"].items()}
        for rec in P.reachable_records()[site]:
            if dict(rec).get(label) == null:
                pol.setdefault(site, {})[rec] = uninformed_action
        return {**sol, "policy": pol}

    return Transformation(f"NULL_AS_ABSENCE[{site}.{label}]", frozenset({"sound", "complete"}), applies, obligations, forward, lift)


# =====================================================================================  world route -> record route
def _all_action_paths(P, omega):
    """Every realizable action assignment (all actions explored), respecting the generator."""
    out = []
    def go(i, acts):
        if i == len(P.events):
            out.append(acts); return
        e = P.events[i]
        if e.exists is not None and not e.exists(omega, acts):
            go(i + 1, acts); return
        site = P.sites[e.site]
        acts_avail = site.actions if site.program is None else [site.program(P.record(e, omega, acts, {}, {}))]
        for a in acts_avail:
            go(i + 1, {**acts, e.name: a})
    go(0, {})
    return out


def _find_source(P, event, label):
    ev = next(e for e in P.events if e.name == event)
    rd = next(r for r in ev.reads if r.label == label and r.kind == "obs")
    for src in [e.name for e in P.events[:P.events.index(ev)]]:
        table, good = {}, True
        for _, om in P.world:
            for acts in _all_action_paths(P, om):
                if src not in acts:
                    continue
                v = rd.fn(om, acts)
                if table.setdefault(acts[src], v) != v:
                    good = False; break
            if not good:
                break
        if good and table:
            return src, table
    return None, None


def _world_obs_obligations(event, label):
    def noise_free(P):
        src, _ = _find_source(P, event, label)
        return ok(src is not None, f"observation is a deterministic function of {src}'s action",
                  "observation is not a deterministic function of any single earlier action (noise or confounding)")

    def sole_reader(P):
        readers = [e.name for e in P.events for r in e.reads if r.kind == "obs" and r.label == label]
        return ok(readers == [event], "no other observer of this world item", f"other observers: {readers}")
    return [Obligation("noise-free, single-source observation", S, noise_free),
            Obligation("no other observers of the world item", S, sole_reader)]


def _swap_read(P, event, label):
    src, table = _find_source(P, event, label)
    keyed = {repr(k): v for k, v in table.items()}
    new_events = []
    for e in P.events:
        if e.name == event:
            reads = tuple(Read(r.label, "msg", (lambda a, _t=keyed: _t.get(repr(a), ABSENT)), src=src) if r.label == label else r
                          for r in e.reads)
            e = replace(e, reads=reads)
        new_events.append(e)
    return src, P.copy(events=new_events)


def _applies_obs(event, label):
    return lambda P: ok(getattr(P, "kind", None) == "finite" and any(e.name == event and any(r.label == label and r.kind == "obs" for r in e.reads) for e in P.events),
                        "event observes the world item", "no such world observation")


def OBS_AS_RECORD(event, label):
    """Re-express a world observation as a record transfer from the acting event. Exact: records are unchanged.
    Needs only a noise-free single source and no other observer; payoff side effects stay in the payoff."""
    return Transformation(f"OBS_AS_RECORD[{event}.{label}]", frozenset({"sound", "complete", "full"}), _applies_obs(event, label),
                          lambda P: _world_obs_obligations(event, label), lambda P: _swap_read(P, event, label)[1])


def WORLD_CHANNEL_AS_MESSAGE(event, label):
    """Stronger claim: the world channel is equivalent to a pure message whose only world effect is an additive cost.
    Needs OBS_AS_RECORD's conditions plus separability of the acting event's payoff effect."""
    def separable(P):
        src, _ = _find_source(P, event, label)
        if src is None:
            return False, "no source action identified"
        for _, om in P.world:
            diffs = {}
            for acts in _all_action_paths(P, om):
                if src not in acts:
                    continue
                others = tuple(sorted((k, repr(v)) for k, v in acts.items() if k != src))
                for o, f in P.payoffs.items():
                    diffs.setdefault((o, repr(acts[src])), {})[others] = f(om, acts)
            for o in P.payoffs:
                keys = [k for k in diffs if k[0] == o]
                for k1, k2 in itertools.combinations(keys, 2):
                    d1, d2 = diffs[k1], diffs[k2]
                    if len({round(d1[c] - d2[c], 12) for c in set(d1) & set(d2)}) > 1:
                        return False, f"the world effect of {src}'s action on {o}'s payoff interacts with other decisions"
        return True, "the action's payoff effect is additive and does not interact with other decisions"

    def forward(P):
        src, Q = _swap_read(P, event, label)
        # rebuild payoffs explicitly as (effect-free part) + (cost of the message): identical when separable
        def make(f):
            def g(om, acts):
                if src not in acts:
                    return f(om, acts)
                ref = P.sites[next(e.site for e in P.events if e.name == src)].actions[0]
                base = f(om, {**acts, src: ref})
                paths = [a for a in _all_action_paths(P, om) if src in a]
                a0 = paths[0]
                cost = f(om, {**a0, src: acts[src]}) - f(om, {**a0, src: ref})
                return base + cost
            return g
        return Q.copy(payoffs={o: make(f) for o, f in P.payoffs.items()})

    return Transformation(f"WORLD_CHANNEL_AS_MESSAGE[{event}.{label}]", frozenset({"sound", "complete", "full"}), _applies_obs(event, label),
                          lambda P: _world_obs_obligations(event, label) + [Obligation("separable side effect", S, separable)], forward)


# =====================================================================================  branching
def FAN_OUT_ACTIONS(site, blocks):
    """Fan out on which block of a site's actions is used; recombine by selecting the best branch.
    For an argmax task, the selected branch optimum is a solution of P only if the branches cover P's policy class,
    so coverage is needed for SOUNDNESS as well as completeness (found by derivation search, F8)."""
    def applies(P):
        good, why = is_finite_team(P)
        return (good and site in P.sites), (why if not good else "site present")

    def covers(P):
        n = len(P.reachable_records()[site])
        if P.sites[site].coin is not None:
            return False, "the site can randomize: a mixed or behavioural policy uses several blocks at one record"
        return ok(n == 1, "single record, no randomization: a policy-level split equals the whole policy class",
                  f"the site has {n} records: branches exclude policies that use different blocks for different records")

    def obligations(P):
        return [Obligation("single-owner team", S, team_ok),
                Obligation("blocks partition the action set", S,
                           lambda P: ok(sorted(map(repr, (a for b in blocks for a in b))) == sorted(map(repr, P.sites[site].actions)), "partition", "not a partition")),
                Obligation("branch policy classes cover the original", S, covers, required=False, for_property="sound"),
                Obligation("branch policy classes cover the original ", S, covers, required=False, for_property="complete")]

    def forward(P):
        return [P.copy(sites={**P.sites, site: replace(P.sites[site], actions=tuple(b))}) for b in blocks]

    def lift(P, branch_solutions):
        best = max((s for sols in branch_solutions for s in sols), key=lambda s: s["value"])
        return [best]

    return Transformation(f"FAN_OUT_ACTIONS[{site}]", frozenset({"sound", "complete"}), applies, obligations, forward, lift, fanout=True)


# =====================================================================================  games
@dataclass
class Bimatrix:
    owners: tuple
    rows: list
    cols: list
    A: np.ndarray
    B: np.ndarray
    origin: object = None
    kind: str = "bimatrix"

    def equivalent(self, s, t):
        """Nash on a normal form: a mixed profile is its own identity (p and q equal)."""
        return bool(np.allclose(s["p"], t["p"], atol=1e-8) and np.allclose(s["q"], t["q"], atol=1e-8))


def PBE_TO_NASH():
    """If world noise gives every information set positive probability under every profile, PBE = Nash."""
    def applies(P):
        return ok(getattr(P, "kind", None) == "finite" and P.concept == "pbe", "PBE problem", "not a PBE problem")

    def full_support(P):
        dsites = {s.name for s in P.decision_sites()}
        free = {s: "uniform" for s in P.sites}
        for i, e in enumerate(P.events):
            if e.site not in dsites:
                continue
            by_hist, allv = {}, set()
            for pw, om in P.world:
                for p, acts, recs, _ in P.paths(free, om):
                    if e.name not in recs:
                        continue
                    hist = tuple((k, repr(v)) for k, v in acts.items() if k in [x.name for x in P.events[:i]])
                    by_hist.setdefault(hist, set()).add(recs[e.name]); allv.add(recs[e.name])
            for h, vals in by_hist.items():
                if vals != allv:
                    return False, f"at {e.name}, some records have zero probability after action history {h}: off-path beliefs unconstrained"
        return True, "every record at every decision event has positive probability after every action history"

    def obligations(P):
        return [Obligation("full-support observations", S, full_support)]

    return Transformation("PBE_TO_NASH", frozenset({"sound", "full"}), applies, obligations, lambda P: P.copy(concept="nash"))


def NORMAL_FORM():
    def applies(P):
        return ok(getattr(P, "kind", None) == "finite" and P.concept == "nash" and len(P.payoffs) == 2,
                  "two-owner Nash problem", "not a two-owner Nash problem")

    def obligations(P):
        return [Obligation("execution semantics well-founded", S, lambda P: P.well_founded())]

    def forward(P):
        o1, o2 = list(P.payoffs)
        s1 = [s.name for s in P.decision_sites() if s.owner == o1]
        s2 = [s.name for s in P.decision_sites() if s.owner == o2]
        R, C = P.pure_policies(s1), P.pure_policies(s2)
        A, B = np.zeros((len(R), len(C))), np.zeros((len(R), len(C)))
        for i, r in enumerate(R):
            for j, c in enumerate(C):
                v = P.value({**r, **c}); A[i, j] = v[o1]; B[i, j] = v[o2]
        return Bimatrix((o1, o2), R, C, A, B, P)

    def lift(P, sol):
        return sol   # mixed profiles over owners' pure policies are already solutions of P

    return Transformation("NORMAL_FORM", frozenset({"sound", "full"}), applies, obligations, forward, lift)


def is_bimatrix(G):
    return ok(getattr(G, "kind", None) == "bimatrix", "bimatrix game", "not a bimatrix game")


def _expand(G, keep_r, keep_c, sol):
    p = np.zeros(len(G.rows)); q = np.zeros(len(G.cols))
    p[keep_r] = sol["p"]; q[keep_c] = sol["q"]
    return {**sol, "p": p, "q": q, "rows": G.rows, "cols": G.cols}


def IESDS():
    """Iterated elimination of strictly dominated pure strategies (both players)."""
    def reduce(G):
        r, c = list(range(len(G.rows))), list(range(len(G.cols)))
        changed = True
        while changed:
            changed = False
            for i in list(r):
                if any(np.all(G.A[k, c] > G.A[i, c] + TOL) for k in r if k != i):
                    r.remove(i); changed = True
            for j in list(c):
                if any(np.all(G.B[r, k] > G.B[r, j] + TOL) for k in c if k != j):
                    c.remove(j); changed = True
        return r, c

    def forward(G):
        r, c = reduce(G)
        H = Bimatrix(G.owners, [G.rows[i] for i in r], [G.cols[j] for j in c], G.A[np.ix_(r, c)], G.B[np.ix_(r, c)], G)
        H._keep = (r, c)
        return H

    def lift(G, sol):
        r, c = reduce(G)
        return _expand(G, r, c, sol)

    def unprofitable(G, s):
        uA, uB = G.A @ s["q"], s["p"] @ G.B
        va, vb = s["p"] @ G.A @ s["q"], s["p"] @ G.B @ s["q"]
        return ok(uA.max() <= va + 1e-7 and uB.max() <= vb + 1e-7, "no removed strategy is a profitable deviation",
                  "a removed strategy is a profitable deviation")

    def obligations(G):
        return [Obligation("iterated strict dominance preserves the Nash set", L, lambda G: (True, "standard"),
                           lemma="strictly dominated strategies are never played in any Nash equilibrium"),
                Obligation("removed strategies unprofitable against the solution", V, unprofitable)]

    return Transformation("IESDS", frozenset({"sound", "full"}), is_bimatrix, obligations, forward, lift)


def IEWDS(player, s, t, strict=False):
    """Remove strategy s of `player` (0=row,1=col) weakly (or, with strict=True, strictly) dominated by t.
    Weak: sound and existence-complete, NOT full. Strict: sound and full."""
    def idx(G):
        labs = G.rows if player == 0 else G.cols
        return (labs.index(s) if s in labs else None), (labs.index(t) if t in labs else None)

    def applies(G):
        a, b = idx(G) if getattr(G, "kind", None) == "bimatrix" else (None, None)
        return ok(a is not None and b is not None, "both strategies present", "strategy not present")

    def dominated(G):
        a, b = idx(G)
        d = (G.A[b, :] - G.A[a, :]) if player == 0 else (G.B[:, b] - G.B[:, a])
        if strict:
            return ok(np.all(d > TOL), f"{s} strictly dominated by {t}", f"{s} is not strictly dominated by {t}")
        return ok(np.all(d >= -TOL) and np.any(d > TOL), f"{s} weakly dominated by {t}", f"{s} is not weakly dominated by {t}")

    def forward(G):
        a, _ = idx(G)
        r = [i for i in range(len(G.rows)) if not (player == 0 and i == a)]
        c = [j for j in range(len(G.cols)) if not (player == 1 and j == a)]
        H = Bimatrix(G.owners, [G.rows[i] for i in r], [G.cols[j] for j in c], G.A[np.ix_(r, c)], G.B[np.ix_(r, c)], G)
        return H

    def lift(G, sol):
        a, _ = idx(G)
        r = [i for i in range(len(G.rows)) if not (player == 0 and i == a)]
        c = [j for j in range(len(G.cols)) if not (player == 1 and j == a)]
        return _expand(G, r, c, sol)

    return Transformation(f"{'ISDS' if strict else 'IEWDS'}[{player}:{s}<{t}]",
                          frozenset({"sound", "complete", "full"} if strict else {"sound", "complete"}), applies,
                          lambda G: [Obligation("strict dominance" if strict else "weak dominance", S, dominated)], forward, lift)


def SUPPORT_ENUM(max_support=4):
    """Terminal: Nash equilibria by support enumeration (duplicates of payoff-identical strategies merged)."""
    def forward(G):
        def dedupe(M1, M2, axis):
            keys, keep = {}, []
            n = M1.shape[axis]
            for i in range(n):
                v = (np.round(M1.take(i, axis), 10).tobytes(), np.round(M2.take(i, axis), 10).tobytes())
                if v not in keys:
                    keys[v] = i; keep.append(i)
            return keep
        rk = dedupe(G.A, G.B, 0); ck = dedupe(G.A, G.B, 1)
        A, B = G.A[np.ix_(rk, ck)], G.B[np.ix_(rk, ck)]
        n, m = A.shape
        found = []
        for k in range(1, min(n, m, max_support) + 1):
            for sr in itertools.combinations(range(n), k):
                for sc in itertools.combinations(range(m), k):
                    M = np.zeros((k + 1, k + 1)); M[:k, :k] = A[np.ix_(sr, sc)]; M[:k, k] = -1; M[k, :k] = 1
                    N_ = np.zeros((k + 1, k + 1)); N_[:k, :k] = B[np.ix_(sr, sc)].T; N_[:k, k] = -1; N_[k, :k] = 1
                    rhs = np.zeros(k + 1); rhs[k] = 1
                    try:
                        q = np.linalg.solve(M, rhs)[:k]; p = np.linalg.solve(N_, rhs)[:k]
                    except np.linalg.LinAlgError:
                        continue
                    if (q < -1e-9).any() or (p < -1e-9).any():
                        continue
                    pf = np.zeros(n); pf[list(sr)] = p; qf = np.zeros(m); qf[list(sc)] = q
                    if (A @ qf).max() > (A @ qf)[list(sr)].max() + 1e-9 or (pf @ B).max() > (pf @ B)[list(sc)].max() + 1e-9:
                        continue
                    P_ = np.zeros(len(G.rows)); Q_ = np.zeros(len(G.cols)); P_[rk] = pf; Q_[ck] = qf
                    key = (tuple(np.round(P_, 8)), tuple(np.round(Q_, 8)))
                    if key not in [f[0] for f in found]:
                        found.append((key, {"p": P_, "q": Q_, "vA": float(pf @ A @ qf), "vB": float(pf @ B @ qf),
                                           "rows": G.rows, "cols": G.cols}))
        return Explicit([f[1] for f in found], note=f"{len(found)} equilibria")

    def is_nash(G, s):
        if s is None:
            return False, "no equilibrium found"
        uA, uB = G.A @ s["q"], s["p"] @ G.B
        return ok(uA.max() <= s["p"] @ G.A @ s["q"] + 1e-7 and uB.max() <= s["p"] @ G.B @ s["q"] + 1e-7,
                  "best-response check passed", "not a Nash equilibrium")

    def obligations(G):
        return [Obligation("every returned profile is a Nash equilibrium", V, lambda G, s: is_nash(G, s) if s is not None else (True, "")),
                Obligation("at least one equilibrium found", V, lambda G, s: ok(s is not None, "found", "none found"),
                           required=False, for_property="complete")]

    def cost(G):
        from math import comb
        n, m = G.A.shape
        return sum(comb(n, k) * comb(m, k) for k in range(1, min(n, m, max_support) + 1))

    return Transformation("SUPPORT_ENUM", frozenset({"sound", "complete"}), is_bimatrix, obligations, forward, terminal=True, cost=cost)


# =====================================================================================  interpretation helpers
def behavioural_marginal(profile, side_policies, site, record, action):
    """P(site plays `action` at `record`) under a mixed profile over pure policies."""
    return float(sum(w for w, pol in zip(profile, side_policies) if pol[site].get(record) == action))
