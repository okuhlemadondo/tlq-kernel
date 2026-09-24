"""v0.3 behaviours found by the blind case (fast tests only; the blind runs themselves live in tests/blind/)."""
import itertools
import sys
import pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent / "blind"))
from blind_case import blind
from cases import weak_dominance_game, parent_child
import tlq.search as SE
from tlq.kernel import Transformation, Explicit, Obligation
from tlq.rules_finite import PURE_NASH, is_finite_team, TEAM_ENUM


def test_f14_ill_formed_problem_is_refused_before_search():
    rep = SE.search(blind("team"))                     # the pre-registered encoding: both types labelled 't'
    assert rep.status == "ILL_FORMED" and "denotes different items" in rep.reasons[0]
    assert blind("team", distinct_labels=True).validate()[0]


def test_f13_refutation_requires_a_certified_bypass():
    """A bypass solver whose own a-posteriori soundness check fails cannot refute anything."""
    P = parent_child()
    bad = Transformation("SHAKY_SOLVER", frozenset({"sound"}), is_finite_team,
                         lambda P: [Obligation("block", "S", lambda P: (False, "always blocked")),
                                    Obligation("own optimality check", "V", lambda P, s: (False, "not certified"),
                                               required=False, for_property="sound")],
                         lambda P: Explicit([{"policy": TEAM_ENUM().forward(P).solutions[0]["policy"], "value": 0.0}]), terminal=True)
    verdict, detail = SE.diagnose_step(bad, P, "check", SE.default_candidates, 2e5)
    assert verdict == "UNESTABLISHED" and "not certified" in detail


def test_pure_nash_agrees_with_brute_force():
    G = weak_dominance_game()
    sols = PURE_NASH().forward(G).solutions
    brute = [(i, j) for i, j in itertools.product(range(3), range(2))
             if G.A[i, j] >= G.A[:, j].max() and G.B[i, j] >= G.B[i, :].max()]
    assert sorted((int(s["p"].argmax()), int(s["q"].argmax())) for s in sols) == sorted(brute)


def test_status_is_registry_and_budget_relative():
    rep = SE.search(parent_child(), max_nodes=50, cost_budget=1e4)
    assert rep.params["cost_budget"] == 1e4 and rep.params["max_nodes"] == 50 and rep.params["fingerprint"]
