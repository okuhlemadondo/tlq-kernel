"""The rule registry: generic instantiators keyed on problem KIND and STRUCTURE, never on problem identity."""
from __future__ import annotations
import itertools
from . import liquidation as LQ
from . import rules_finite as RF


def liquidation_rules(P):
    if getattr(P, "kind", None) != "liquidation":
        return []
    return [LQ.SHIFT, LQ.ADAPT, LQ.CONST_FORECAST, LQ.OPEN_LOOP, LQ.SCALE, LQ.SOLVE_RECURRENCE, LQ.SOLVE_QP, LQ.CE_MARTINGALE]


def finite_solvers(P):
    if getattr(P, "kind", None) != "finite":
        return []
    return [RF.TEAM_ENUM(), RF.BEHAVIOURAL_OPT(), RF.COMMON_INFO(), RF.PBE_TO_NASH(), RF.NORMAL_FORM()]


def finite_rewrites(P):
    """One instance per structural opportunity: every world observation, every declared-relevant record item, every site."""
    if getattr(P, "kind", None) != "finite":
        return []
    out = []
    for e in P.events:
        for r in e.reads:
            if r.kind == "obs":
                out += [RF.OBS_AS_RECORD(e.name, r.label), RF.WORLD_CHANNEL_AS_MESSAGE(e.name, r.label)]
    recs = P.reachable_records()
    for s in P.decision_sites():
        if s.name in P.relevant:
            items = {}
            for rec in recs[s.name]:
                if not s.memory:
                    for lab, v in rec:
                        items.setdefault(lab, set()).add(v)
            for lab, vals in items.items():
                for v in sorted(vals, key=repr):
                    for a in s.actions:
                        out.append(RF.NULL_AS_ABSENCE(s.name, lab, v, a))
        acts = list(s.actions)
        for k in range(1, len(acts) // 2 + 1):
            for blk in itertools.combinations(acts, k):
                rest = tuple(a for a in acts if a not in blk)
                if k < len(acts) - k or blk < rest:
                    out.append(RF.FAN_OUT_ACTIONS(s.name, [tuple(blk), rest]))
    return out


def coordinator_rules(P):
    return [RF.COORDINATOR_DP()] if getattr(P, "kind", None) == "coordinator" else []


def game_rules(P, weak_dominance_budget=36):
    if getattr(P, "kind", None) != "bimatrix":
        return []
    out = [RF.IESDS(), RF.SUPPORT_ENUM(), RF.PURE_NASH()]
    if len(P.rows) * len(P.cols) <= weak_dominance_budget:        # a resource bound, not problem knowledge
        out += [RF.IEWDS(0, s, t) for s, t in itertools.permutations(P.rows, 2)]
        out += [RF.IEWDS(1, s, t) for s, t in itertools.permutations(P.cols, 2)]
    return out


DEFAULT = [liquidation_rules, finite_solvers, finite_rewrites, coordinator_rules, game_rules]


def candidates(P, registry=DEFAULT):
    return [rule for inst in registry for rule in inst(P)]
