"""Finite problems P = (W, E, T) for the prototype.

W  world         : a finite distribution over outcomes omega (dicts). Observation items read world facts,
                   which may depend on realized actions (realization -> world -> observation).
E  execution     : sites (program sites with a record store), events (evaluations of a site),
                   the generator Gamma (event existence predicates on the unfolding), reads
                   (world observations vs record transfers A(src, site)), and execution semantics X
                   (events are evaluated in list order; a read may only reference earlier events).
T  task          : owners' payoffs on (omega, actions) and a solution concept.

Key semantic choices (each forced by a regression case):
  * a POLICY belongs to a SITE and reads the site's RECORD; two evaluations of the same site with the same
    record act identically. Evaluations of different sites need not.            (driver case)
  * `memory=True` means A(n,n) carries the site's whole history, including a visit counter (perfect recall).
  * a record transfer from an event that does not exist delivers ABSENT; the read map decides how absence
    appears to the receiver.                                                    (endogenous absence)
  * randomness lives somewhere: coin='fresh' (new private draw each evaluation, not carried forward:
    behavioural), 'persistent' (one draw per site, carried forward: mixtures of pure plans), None (pure).
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable
import itertools

ABSENT = "<absent>"


@dataclass(frozen=True)
class Site:
    name: str
    owner: str
    actions: tuple
    memory: bool = False
    coin: str | None = None
    program: Callable | None = None      # fixed program record -> action (not a decision variable)


@dataclass(frozen=True)
class Read:
    label: str
    kind: str                     # 'obs' (world route) | 'msg' (record route)
    fn: Callable                  # obs: fn(omega, acts) ; msg: fn(value of source action/record or ABSENT)
    src: str | None = None        # msg: source event
    of: str = "action"            # msg: transfer the source event's 'action' or its 'record'


@dataclass(frozen=True)
class Event:
    name: str
    site: str
    reads: tuple = ()
    exists: Callable = None       # Gamma: (omega, acts) -> bool ; None = always


@dataclass
class FiniteProblem:
    world: list                   # [(prob, omega)]
    sites: dict                   # name -> Site
    events: list                  # [Event] in execution order
    payoffs: dict                 # owner -> fn(omega, acts)
    concept: str                  # 'team' | 'nash' | 'pbe'
    relevant: dict = field(default_factory=dict)       # site -> fn(omega): the task-relevant state for that site
    restrictions: list = field(default_factory=list)   # [(site, predicate(record)->bool, forced action)]
    kind: str = "finite"

    def copy(self, **kw):
        return replace(self, **kw)

    # ------------------------------------------------------------ execution semantics X
    def well_founded(self):
        pos = {e.name: i for i, e in enumerate(self.events)}
        for i, e in enumerate(self.events):
            for r in e.reads:
                if r.kind == "msg" and (r.src not in pos or pos[r.src] >= i):
                    return False, f"event {e.name} reads {r.src}, which is not strictly earlier"
        return True, "every record transfer points to an earlier event"

    def decision_sites(self):
        return [s for s in self.sites.values() if s.program is None and len(s.actions) > 1]

    def record(self, ev, omega, acts, recs, hist):
        cur = []
        for r in ev.reads:
            if r.kind == "obs":
                cur.append((r.label, r.fn(omega, acts)))
            else:
                src_val = (acts if r.of == "action" else recs).get(r.src, ABSENT)
                cur.append((r.label, r.fn(src_val)))
        cur = tuple(cur)
        site = self.sites[ev.site]
        if site.memory:
            past = tuple(hist.get(site.name, ()))
            return (len(past), past, cur)
        return cur

    def _choice(self, site, rec, policy):
        for (sname, pred, act) in self.restrictions:
            if sname == site.name and pred(rec):
                return {act: 1.0}
        if site.program is not None:
            a = site.program(rec)
        elif len(site.actions) == 1:
            a = site.actions[0]
        else:
            pol = policy.get(site.name, {})
            if pol == "uniform":
                return {x: 1.0 / len(site.actions) for x in site.actions}
            a = pol.get(rec, site.actions[0])
        return a if isinstance(a, dict) else {a: 1.0}

    def paths(self, policy, omega):
        """Yield (prob, acts, recs, visits) for one world outcome under a (possibly behavioural) policy."""
        def go(i, p, acts, recs, hist, visits):
            if i == len(self.events):
                yield p, acts, recs, visits
                return
            ev = self.events[i]
            if ev.exists is not None and not ev.exists(omega, acts):
                yield from go(i + 1, p, acts, recs, hist, visits)
                return
            site = self.sites[ev.site]
            rec = self.record(ev, omega, acts, recs, hist)
            for a, q in self._choice(site, rec, policy).items():
                if q <= 0:
                    continue
                h2 = dict(hist); h2[site.name] = tuple(hist.get(site.name, ())) + ((rec, a),)
                yield from go(i + 1, p * q, {**acts, ev.name: a}, {**recs, ev.name: rec}, h2,
                              visits + ((site.name, rec),))
        yield from go(0, 1.0, {}, {}, {}, ())

    def value(self, policy):
        out = {o: 0.0 for o in self.payoffs}
        for pw, omega in self.world:
            for p, acts, _, _ in self.paths(policy, omega):
                for o, f in self.payoffs.items():
                    out[o] += pw * p * f(omega, acts)
        return out

    # ------------------------------------------------------------ solution concept: its declared equivalence
    def outcome_distribution(self, sol):
        """Distribution over complete histories (world outcome index, actions) induced by a solution."""
        if "policy" in sol:
            profiles = [(1.0, sol["policy"])]
        else:
            profiles = [(pi * qj, {**r, **c}) for pi, r in zip(sol["p"], sol["rows"]) if pi > 1e-12
                        for qj, c in zip(sol["q"], sol["cols"]) if qj > 1e-12]
        dist = {}
        for w, pol in profiles:
            for k, (pw, om) in enumerate(self.world):
                for p, acts, _, _ in self.paths(pol, om):
                    key = (k, tuple(sorted((e, repr(a)) for e, a in acts.items())))
                    dist[key] = dist.get(key, 0.0) + w * pw * p
        return dist

    def behaviour(self, sol):
        """Behavioural strategy at every reachable record of every decision site (marginals over pure policies)."""
        if "policy" in sol:
            return {(s, repr(r)): repr(a) for s, pol in sol["policy"].items() if pol != "uniform" for r, a in pol.items()}
        out = {}
        for w, pols in ((sol["p"], sol["rows"]), (sol["q"], sol["cols"])):
            for x, pol in zip(w, pols):
                for s, rules in pol.items():
                    for r, a in rules.items():
                        k = (s, repr(r), repr(a))
                        out[k] = out.get(k, 0.0) + x
        return out

    def equivalent(self, s, t, tol=1e-9):
        """The solution concept declares its own equivalence:
           team / nash : realization (outcome) equivalence -- same distribution over complete histories;
           pbe         : finer -- same behaviour at every record, on and off path."""
        if self.concept == "pbe":
            a, b = self.behaviour(s), self.behaviour(t)
        else:
            a, b = self.outcome_distribution(s), self.outcome_distribution(t)
        return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= tol for k in set(a) | set(b))

    # ------------------------------------------------------------ reachable records (all actions explored)
    def reachable_records(self):
        found = {s.name: set() for s in self.sites.values()}
        for _, omega in self.world:
            def go(i, acts, recs, hist):
                if i == len(self.events):
                    return
                ev = self.events[i]
                if ev.exists is not None and not ev.exists(omega, acts):
                    go(i + 1, acts, recs, hist); return
                site = self.sites[ev.site]
                rec = self.record(ev, omega, acts, recs, hist)
                found[site.name].add(rec)
                for a in self._choice(site, rec, {site.name: "uniform"}):
                    h2 = dict(hist); h2[site.name] = tuple(hist.get(site.name, ())) + ((rec, a),)
                    go(i + 1, {**acts, ev.name: a}, {**recs, ev.name: rec}, h2)
            go(0, {}, {}, {})
        return {k: sorted(v, key=repr) for k, v in found.items()}

    def pure_policies(self, sites=None):
        """All pure policies of the given decision sites: list of {site: {record: action}}."""
        recs = self.reachable_records()
        sites = sites if sites is not None else [s.name for s in self.decision_sites()]
        per_site = []
        for sn in sites:
            rs = recs[sn]
            per_site.append([dict(zip(rs, combo)) for combo in itertools.product(self.sites[sn].actions, repeat=len(rs))])
        return [dict(zip(sites, combo)) for combo in itertools.product(*per_site)]

    # ------------------------------------------------------------ derived structure
    def repeats_site_record(self):
        """Is some (site, record) evaluated more than once on some path? (fails multilinearity)"""
        free = {s: "uniform" for s in self.sites}
        for _, omega in self.world:
            for _, _, _, visits in self.paths(free, omega):
                if len(set(visits)) < len(visits):
                    return True
        return False

    def generator_depends_on_actions(self):
        """Does Gamma make any event's existence depend on realized actions? (endogenous structure)"""
        free = {s: "uniform" for s in self.sites}
        for ev in self.events:
            if ev.exists is None:
                continue
            for _, omega in self.world:
                vals = {ev.exists(omega, acts) for _, acts, _, _ in self.paths(free, omega)}
                if len(vals) > 1:
                    return True
        return False
