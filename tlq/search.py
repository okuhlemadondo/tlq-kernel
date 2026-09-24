"""Derivation search over the kernel interfaces.

The searcher is given a root problem and a registry. It never sees problem identities.
Every edge it keeps was accepted by the kernel (applicability + S/L/D obligations); every terminal it keeps
survived the kernel's a-posteriori (V) obligations. It distinguishes:

  FOUND                       at least one certified derivation reached a solved class
  REGISTRY_GAP                a reachable problem had no applicable rule at all (vocabulary/solved-class gap)
  NOT_ESTABLISHED             blocked steps exist that are not refuted: a derivation may exist that the
                              certificate system cannot establish (certification is conservative)
  NO_DERIVATION_IN_REGISTRY   every blocked step was refuted and nothing was left unexplored
  (+ exhausted flag)          the node/cost budget cut the search: absence of a derivation is not established

Blocked steps are diagnosed, when possible, by bypassing the certificate and solving both sides:
  REFUTED        the step demonstrably fails to preserve (a value gap, or a concrete a-posteriori violation)
  CONSERVATIVE   not certified, yet in this instance the step loses nothing (certificates are sufficient, not necessary)
  UNESTABLISHED  neither: no bypass or no comparison is available
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .kernel import Derivation, Rejected, NotExecutable, partition
from .registry import candidates as default_candidates


@dataclass
class Found:
    rules: list
    result: object


@dataclass
class Blocked:
    path: list
    rule: str
    stage: str               # 'check' (S/L/D) | 'solve' (V) | 'branch' | 'budget'
    failures: list
    verdict: str = ""
    detail: str = ""


@dataclass
class Report:
    found: list = field(default_factory=list)
    blocked: list = field(default_factory=list)
    dead_ends: list = field(default_factory=list)
    exhausted: bool = False
    nodes: int = 0
    status: str = ""
    reasons: list = field(default_factory=list)
    terminal_seen: bool = False       # did any explored problem admit a solved class for its solution concept?


def signature(P):
    k = getattr(P, "kind", None)
    if k == "liquidation":
        return (k, P.form, P.Q0)
    if k == "finite":
        return (k, P.concept, tuple((e.name, tuple((r.label, r.kind, r.src) for r in e.reads)) for e in P.events),
                len(P.restrictions), tuple((s.name, s.actions) for s in P.sites.values()))
    if k == "bimatrix":
        return (k, P.A.shape, P.A.tobytes(), P.B.tobytes())
    if k == "coordinator":
        return (k, signature(P.base))
    return (k, id(P))


def search(P0, candidates=default_candidates, max_depth=8, max_nodes=400, cost_budget=2e5,
           all_derivations=True, diagnose=True, max_diagnoses=12):
    rep = Report()

    def dfs(d: Derivation, path, seen, depth):
        if rep.nodes >= max_nodes:
            rep.exhausted = True
            return
        rep.nodes += 1
        P = d.current
        cands = candidates(P)
        applicable = [r for r in cands if r.applies(P)[0]]
        if not applicable:
            rep.dead_ends.append((list(path), getattr(P, "kind", "?")))
            return
        applicable.sort(key=lambda r: (not r.terminal, r.fanout))      # try solved classes first
        if any(r.terminal for r in applicable):
            rep.terminal_seen = True
        for rule in applicable:
            if rep.found and not all_derivations:
                return
            if rule.cost is not None and rule.cost(P) > cost_budget:
                rep.blocked.append(Blocked(list(path), rule.name, "budget", [], "UNESTABLISHED",
                                           f"estimated cost {rule.cost(P):.3g} exceeds budget {cost_budget:.3g}"))
                rep.exhausted = True
                continue
            d2 = d.clone()
            try:
                if rule.fanout:
                    subs = []
                    for Q in rule.forward(P):
                        sub = search(Q, candidates, max_depth - depth - 1, max_nodes // 4, cost_budget, all_derivations=False, diagnose=False)
                        if not sub.found:
                            raise Rejected([(rule.name, "branch", "a branch has no certified derivation")])
                        subs.append(sub.found[0].rules)
                    def plan(i, sd, _subs=subs):
                        for rr in _subs[i]:
                            sd.apply(rr)
                        return sd
                    d2.fan(rule, plan)
                else:
                    d2.apply(rule)
            except Rejected as e:
                if any(o == "applicability" for _, o, _ in e.failures):
                    continue
                rep.blocked.append(Blocked(list(path), rule.name, "branch" if rule.fanout else "check", e.failures))
                rep.blocked[-1]._ctx = (rule, P)
                continue
            except Exception as e:      # a rule that crashes after its obligations passed: an implementation or obligation gap
                rep.blocked.append(Blocked(list(path), rule.name, "error", [(rule.name, type(e).__name__, str(e)[:120])],
                                           "IMPLEMENTATION_ERROR", "the rule was admitted but its map failed"))
                rep.blocked[-1]._ctx = (rule, P)
                continue
            if rule.terminal or rule.fanout:
                try:
                    res = d2.solve()
                    if "sound" in res.preserves:
                        rep.found.append(Found(path + [rule], res))
                    else:
                        rep.blocked.append(Blocked(list(path), rule.name, "uncertified", [],
                                                   "", "reached a solved class, but the derivation certifies no soundness"))
                        rep.blocked[-1]._ctx = (rule, P, res)
                except Rejected as e:
                    rep.blocked.append(Blocked(list(path), rule.name, "solve", e.failures))
                    rep.blocked[-1]._ctx = (rule, P)
                except NotExecutable:
                    pass
                continue
            if depth + 1 >= max_depth:
                rep.exhausted = True
                continue
            sig = signature(d2.current)
            if sig in seen:
                continue                     # no progress: the rule returned an already-visited problem
            dfs(d2, path + [rule], seen | {sig}, depth + 1)

    dfs(Derivation(P0), [], {signature(P0)}, 0)

    if diagnose:
        n = 0
        for b in rep.blocked:
            if b.stage in ("check", "solve") and n < max_diagnoses and hasattr(b, "_ctx"):
                n += 1
                b.verdict, b.detail = diagnose_step(*b._ctx, stage=b.stage, candidates=candidates, cost_budget=cost_budget)
            elif b.stage == "uncertified" and n < max_diagnoses and hasattr(b, "_ctx"):
                n += 1
                rule, P, res = b._ctx
                cert = best_value(P, candidates, cost_budget)
                vals = [v for v in (evaluate(P, x) for x in res.solutions) if v is not None]
                if cert is None or not vals:
                    b.verdict, b.detail = "UNESTABLISHED", "no comparison available"
                elif max(vals) < cert - 1e-9:
                    b.verdict, b.detail = "REFUTED", f"the uncertified result loses value: {max(vals):.6g} < certified {cert:.6g}"
                else:
                    b.verdict, b.detail = "CONSERVATIVE", f"uncertified, but nothing lost here: {max(vals):.6g} vs {cert:.6g}"
            elif not b.verdict:
                b.verdict = "UNESTABLISHED"
    rep.status, rep.reasons = classify(rep, P0)
    return rep


def classify(rep, P0=None):
    """Never read an unrefuted blocked step as non-existence."""
    if rep.found:
        return "FOUND", (["budget cut part of the search"] if rep.exhausted else [])
    reasons = []
    if not rep.terminal_seen:
        reasons.append(f"no solved class in the registry for solution concept '{getattr(P0, 'concept', getattr(P0, 'kind', '?'))}' "
                       "at any reachable problem")
    if rep.dead_ends:
        reasons.append(f"{len(rep.dead_ends)} reachable problem(s) with no applicable rule")
    if rep.exhausted:
        reasons.append("node or cost budget exhausted")
    real = [b for b in rep.blocked if b.stage not in ("budget",)]
    unrefuted = [b for b in real if b.verdict != "REFUTED"]
    if unrefuted:
        reasons.append(f"{len(unrefuted)} blocked step(s) not refuted (certificate not established, or not diagnosed)")
    if not real and rep.dead_ends and rep.terminal_seen is False:
        return "REGISTRY_GAP", reasons
    if reasons:
        return "NOT_ESTABLISHED", reasons
    return "NO_DERIVATION_IN_REGISTRY", ["every blocked step was refuted; nothing left unexplored"]


# ----------------------------------------------------------------------------------- diagnosis
def evaluate(P, sol):
    """Value of a solution in problem P, when the problem class defines one; None otherwise."""
    k = getattr(P, "kind", None)
    if k == "finite" and P.concept == "team" and "policy" in sol:
        for site, pol in sol["policy"].items():
            if pol == "uniform" or site not in P.sites:
                continue
            for d in pol.values():
                if isinstance(d, dict) and max(d.values()) < 1 - 1e-6 and P.sites[site].coin is None:
                    return None           # a randomized policy is not admissible where there is no randomness
        return P.value(sol["policy"])[next(iter(P.payoffs))]
    if k == "liquidation" and "Q" in sol and P.form in ("deterministic", "known_forecast"):
        from .liquidation import objective
        mu = P.context.get("mu", 0.0)
        a = -np.diff(sol["Q"])
        if np.any(a < -1e-9) or np.any(sol["Q"] < -1e-9):
            return -np.inf
        return objective(P, sol["Q"], mu)
    return None


def best_value(P, candidates, cost_budget):
    rep = search(P, candidates, max_depth=6, max_nodes=120, cost_budget=cost_budget, all_derivations=True, diagnose=False)
    vals = [evaluate(P, s) for f in rep.found for s in f.result.solutions]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def diagnose_step(rule, P, stage, candidates, cost_budget):
    if stage == "solve":
        return "REFUTED", "an a-posteriori obligation found a concrete violation in the produced solution"
    try:
        Q = rule.forward(P)
    except Exception as e:
        return "UNESTABLISHED", f"no bypass: the forward map is undefined here ({type(e).__name__})"
    certified = best_value(P, candidates, cost_budget)
    if certified is None:
        return "UNESTABLISHED", "no certified value for the problem before the step"
    if rule.terminal:
        bypass_sols = list(Q.solutions)
    else:
        rep = search(Q, candidates, max_depth=6, max_nodes=120, cost_budget=cost_budget, all_derivations=True, diagnose=False)
        bypass_sols = [s for f in rep.found for s in f.result.solutions]
    vals = []
    for s in bypass_sols:
        try:
            v = evaluate(P, s if rule.terminal else rule.lift(P, s))
        except Exception:
            v = None
        if v is not None:
            vals.append(v)
    if not vals:
        return "UNESTABLISHED", "the bypassed problem could not be solved and evaluated in the original"
    bypass = max(vals)
    if bypass < certified - 1e-9:
        return "REFUTED", f"bypassing the certificate loses value: {bypass:.6g} < certified {certified:.6g}"
    return "CONSERVATIVE", f"not certified, yet nothing is lost in this instance: {bypass:.6g} vs certified {certified:.6g}"


# ----------------------------------------------------------------------------------- comparing derivations
def compare(P0, found):
    """Relate the solution sets of different derivations under the task's own equivalence."""
    eq = getattr(P0, "equivalent", None) or (lambda a, b: a is b)
    out = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            A, B = found[i].result.classes, found[j].result.classes
            inA = [any(eq(b[0], a[0]) for a in A) for b in B]
            inB = [any(eq(a[0], b[0]) for b in B) for a in A]
            if all(inA) and all(inB):
                rel = "equal"
            elif all(inA):
                rel = "second ⊂ first"
            elif all(inB):
                rel = "first ⊂ second"
            else:
                rel = "overlap" if any(inA) else "disjoint"
            va = [evaluate(P0, a[0]) for a in A]
            vb = [evaluate(P0, b[0]) for b in B]
            consistent = True
            if all(v is not None for v in va + vb) and va and vb:
                consistent = abs(max(va) - max(vb)) < 1e-6 * max(1.0, abs(max(va)))
            out.append((i, j, rel, consistent))
    return out
