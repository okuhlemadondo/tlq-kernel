"""Derivation search: the searcher gets only the root problem and the registry.

Checks: it finds derivations; it finds more than one where more than one exists; ledgers survive along every
derivation; lifted solutions are compared under the task's own equivalence; invalid branches are detected and
diagnosed; and 'no derivation found' is kept distinct from 'certificate system could not establish one'."""
import inspect
import numpy as np
from cases import (liquidation, parent_child, driver, silence, spawning, weak_dominance_game, composite, channel)
import tlq.search as SE
import tlq.registry as REG
from tlq.kernel import Transformation, Explicit
from tlq.rules_finite import TEAM_ENUM, is_finite_team
from tlq import liquidation as LQ

BAYES = {"forecast": "bayes_gaussian", "m0": 0.0, "s0": 1.0, "sigma": 1.0, "integrable": True}


def names(f):
    return [r.name for r in f.rules]


def blocked(rep, rule_prefix):
    return [b for b in rep.blocked if b.rule.startswith(rule_prefix)]


# ------------------------------------------------------------------ the searcher does not know the cases
def test_searcher_and_registry_contain_no_case_knowledge():
    src = inspect.getsource(SE) + inspect.getsource(REG)
    for token in ("cases", "parent_child", "driver", "silence", "spawning", "composite", "channel", "C1", "C8", "13/20", "0.65"):
        assert token not in src, token


# ------------------------------------------------------------------ C1: three derivations, one solution
def test_c1_search_finds_all_three_derivations_and_they_agree():
    P = liquidation()
    rep = SE.search(P)
    assert rep.status == "FOUND"
    found = {tuple(names(f)) for f in rep.found}
    assert ("SHIFT", "ADAPT", "CONST_FORECAST", "OPEN_LOOP", "SCALE", "SOLVE_RECURRENCE") in found      # the scripted one
    assert ("SHIFT", "ADAPT", "CONST_FORECAST", "OPEN_LOOP", "SOLVE_RECURRENCE") in found
    assert ("SHIFT", "ADAPT", "CONST_FORECAST", "OPEN_LOOP", "SOLVE_QP") in found
    assert all(rel == "equal" and ok for _, _, rel, ok in SE.compare(P, rep.found))
    for f in rep.found:                                     # the ledger is carried along every derivation
        assert {o for _, o in f.result.ledger} == {"increments integrable", "forecast E[xi_t | F_{t-1}] is deterministic"}
        assert all(rec.checked is not None for rec in f.result.trace)


def test_c1_binding_constraints_invalid_branches_are_refuted():
    rep = SE.search(liquidation(context={"forecast": "constant", "mu": 100.0, "integrable": True}))
    assert [names(f)[-1] for f in rep.found] == ["SOLVE_QP"]
    (rec,) = blocked(rep, "SOLVE_RECURRENCE")
    assert rec.stage == "solve" and rec.verdict == "REFUTED"
    (sc,) = blocked(rep, "SCALE")
    assert sc.verdict == "REFUTED" and "loses value" in sc.detail


def test_c1_no_derivation_in_registry_only_when_every_block_is_refuted():
    restricted = lambda P: [r for r in REG.candidates(P) if r.name not in ("SOLVE_QP", "SCALE", "CE_MARTINGALE")]
    rep = SE.search(liquidation(context={"forecast": "constant", "mu": 100.0, "integrable": True}), candidates=restricted)
    assert rep.status == "NO_DERIVATION_IN_REGISTRY"
    assert all(b.verdict == "REFUTED" for b in rep.blocked)


# ------------------------------------------------------------------ C2
def test_c2_search_finds_the_state_reduction_and_rejects_the_constant_forecast():
    rep = SE.search(liquidation(context=BAYES))
    assert [names(f) for f in rep.found] == [["SHIFT", "ADAPT", "CE_MARTINGALE"]]
    (cf,) = blocked(rep, "CONST_FORECAST")
    assert any("contradicts the world model" in why for _, _, why in cf.failures)


# ------------------------------------------------------------------ C3: two valid derivations
def test_c3_two_valid_derivations_consistent_under_the_tasks_equivalence():
    P = parent_child()
    rep = SE.search(P)
    found = {tuple(names(f)): f for f in rep.found}
    assert ("TEAM_ENUM",) in found and ("COMMON_INFO", "COORDINATOR_DP") in found
    assert len(found[("TEAM_ENUM",)].result.classes) == 2          # two genuinely different optimal codes
    assert len(found[("COMMON_INFO", "COORDINATOR_DP")].result.classes) == 1
    rel = {(names(rep.found[i])[0], names(rep.found[j])[0]): (r, ok) for i, j, r, ok in SE.compare(P, rep.found)}
    assert rel[("TEAM_ENUM", "COMMON_INFO")] == ("second ⊂ first", True)     # consistent: neither claims 'full'
    for b in blocked(rep, "FAN_OUT_ACTIONS"):                        # uncertified branches are not counted as found
        assert b.stage == "uncertified" and b.verdict == "REFUTED"


# ------------------------------------------------------------------ C4
def test_c4_search_uses_behavioural_strategies_and_refutes_pure_enumeration():
    rep = SE.search(driver())
    assert [names(f) for f in rep.found] == [["BEHAVIOURAL_OPT"]]
    (te,) = blocked(rep, "TEAM_ENUM")
    assert te.verdict == "REFUTED" and "1 < certified 1.33333" in te.detail


# ------------------------------------------------------------------ C5 / C6: conservative vs refuted
def test_c5_blocked_silence_steps_are_conservative_not_refuted():
    rep = SE.search(silence())
    nulls = blocked(rep, "NULL_AS_ABSENCE")
    assert nulls and all(b.verdict == "CONSERVATIVE" for b in nulls)


def test_c6_blocked_absence_steps_split_into_refuted_and_conservative():
    rep = SE.search(spawning(prior=0.6))
    verdicts = {b.verdict for b in blocked(rep, "NULL_AS_ABSENCE")}
    assert verdicts == {"REFUTED", "CONSERVATIVE"}


# ------------------------------------------------------------------ C7
def test_c7_search_finds_order_dependent_derivations_all_consistent():
    G = weak_dominance_game()
    rep = SE.search(G, diagnose=False)
    sizes = {len(f.result.classes) for f in rep.found}
    assert len(sizes) > 1                                          # different orders keep different equilibria
    assert all(ok for *_, ok in SE.compare(G, rep.found))
    assert not any("full" in f.result.preserves for f in rep.found if len(f.rules) > 1)


# ------------------------------------------------------------------ C8
def test_c8_search_finds_the_round6_derivation_under_a_cost_budget():
    rep = SE.search(composite(), diagnose=False)
    (f,) = rep.found
    assert names(f) == ["PBE_TO_NASH", "NORMAL_FORM", "IESDS", "SUPPORT_ENUM"]
    assert rep.exhausted and any(b.stage == "budget" for b in rep.blocked)     # the unreduced game was too costly
    s = f.result.solution                                          # one class under the PBE equivalence
    pH = sum(w for w, pol in zip(s["p"], s["rows"]) if pol["A"][(("theta", "H"),)] == "lit")
    assert abs(pH - 0.8889) < 1e-4


def test_c8_record_arrow_leak_is_not_established_not_declared_impossible():
    rep = SE.search(composite(leak="arrow"))
    assert rep.status == "NOT_ESTABLISHED"
    assert any("no solved class in the registry for solution concept 'pbe'" in r for r in rep.reasons)
    assert all(b.verdict != "REFUTED" for b in rep.blocked)


# ------------------------------------------------------------------ C9: composition creates access
def test_c9_rewriting_the_world_observation_makes_common_information_reachable():
    rep = SE.search(channel("world", gamma=0.05))
    found = [tuple(names(f)) for f in rep.found]
    assert ("OBS_AS_RECORD[eP.o]", "COMMON_INFO", "COORDINATOR_DP") in found
    (ci,) = [b for b in rep.blocked if b.rule == "COMMON_INFO" and b.path == []]
    assert "no world-mediated propagation between sites" in {o for _, o, _ in ci.failures}
    assert all(ok for *_, ok in SE.compare(channel("world", gamma=0.05), rep.found))


# ------------------------------------------------------------------ registry gap
def test_registry_gap_is_reported_as_such():
    no_solvers = lambda P: [r for r in REG.candidates(P) if r.name not in ("TEAM_ENUM", "BEHAVIOURAL_OPT", "COORDINATOR_DP")]
    rep = SE.search(parent_child(), candidates=no_solvers)
    assert rep.status == "NOT_ESTABLISHED" and rep.dead_ends
    assert any("no solved class in the registry" in r for r in rep.reasons)


# ------------------------------------------------------------------ an unsound rule is caught by the consistency audit
def test_an_unsound_certificate_is_exposed_by_comparing_derivations():
    P = parent_child()

    def fake_forward(P):      # claims 'sound' but silently optimizes over a restricted class (child never signals)
        Q = P.copy(sites={**P.sites, "child": P.sites["child"].__class__("child", "team", (0,))})
        return Explicit(TEAM_ENUM().forward(Q).solutions)
    FAKE = Transformation("FAKE_SOLVER", frozenset({"sound", "complete"}), is_finite_team, lambda P: [], fake_forward, terminal=True)
    rep = SE.search(P, candidates=lambda Q: REG.candidates(Q) + ([FAKE] if getattr(Q, "kind", None) == "finite" else []), diagnose=False)
    rel = SE.compare(P, rep.found)
    fake_i = [i for i, f in enumerate(rep.found) if names(f) == ["FAKE_SOLVER"]][0]
    assert any(not ok for i, j, _, ok in rel if fake_i in (i, j))
