"""C7 non-confluence, C8 round-6 equilibria, C9 world route vs record route (+ branching)."""
import itertools
import numpy as np
import pytest
from cases import weak_dominance_game, strict_dominance_game, composite, channel, route_choice
from tlq.kernel import Derivation, Rejected, confluence
from tlq.rules_finite import (IEWDS, IESDS, SUPPORT_ENUM, PBE_TO_NASH, NORMAL_FORM, OBS_AS_RECORD, WORLD_CHANNEL_AS_MESSAGE, TEAM_ENUM,
                              FAN_OUT_ACTIONS)


# ------------------------------------------------------------------ C7
def all_instances(G, strict=False):
    return [IEWDS(0, s, t, strict) for s, t in itertools.permutations(G.rows, 2)] + \
           [IEWDS(1, s, t, strict) for s, t in itertools.permutations(G.cols, 2)]


def test_c7_weak_dominance_is_not_confluent():
    G = weak_dominance_game()
    normal = confluence(G, all_instances(G), lambda H: (tuple(H.rows), tuple(H.cols)))
    assert len(normal) >= 2
    payoffs = set()
    for rows, cols in normal:
        if len(rows) >= 1 and len(cols) == 1:
            payoffs |= {(G.A[G.rows.index(r), G.cols.index(cols[0])], G.B[G.rows.index(r), G.cols.index(cols[0])]) for r in rows}
    assert (2.0, 1.0) in payoffs and (1.0, 1.0) in payoffs


def test_c7_strict_dominance_is_confluent():
    G = strict_dominance_game()
    assert len(confluence(G, all_instances(G, strict=True), lambda H: (tuple(H.rows), tuple(H.cols)))) == 1


def test_c7_weak_elimination_is_sound_but_not_full():
    G = weak_dominance_game()
    r = Derivation(G).apply(IEWDS(0, "T", "M")).apply(IEWDS(1, "L", "R")).apply(SUPPORT_ENUM()).solve(claim={"sound"})
    for s in r.solutions:                                   # soundness: lifted profiles are equilibria of the ORIGINAL game
        assert (G.A @ s["q"]).max() <= s["p"] @ G.A @ s["q"] + 1e-9
        assert (s["p"] @ G.B).max() <= s["p"] @ G.B @ s["q"] + 1e-9
    with pytest.raises(Rejected) as e:
        Derivation(G).apply(IEWDS(0, "T", "M")).apply(IEWDS(1, "L", "R")).apply(SUPPORT_ENUM()).solve(claim={"full"})
    assert "claim" in e.value.names()


def test_c7_reject_elimination_of_undominated_strategy():
    with pytest.raises(Rejected) as e:
        Derivation(weak_dominance_game()).apply(IEWDS(0, "M", "T"))
    assert "weak dominance" in e.value.names()


# ------------------------------------------------------------------ C8
def behaviour(s):
    """Behavioural summary of a mixed profile over pure site policies (realization-equivalent profiles coincide)."""
    rows, cols = s["rows"], s["cols"]
    f = lambda w, pols, site, rec, act: float(sum(x for x, pol in zip(w, pols) if pol[site].get(rec) == act))
    return (round(f(s["p"], rows, "A", (("theta", "H"),), "lit"), 4), round(f(s["p"], rows, "A", (("theta", "N"),), "lit"), 4),
            round(sum(x for x, pol in zip(s["q"], cols) if pol["B1"].get(next(k for k in pol["B1"] if dict(k).get("o1") == 1)) == 1), 4),
            round(s["vA"], 4), round(s["vB"], 4))


def solve_composite(P, claim=("sound",)):
    return Derivation(P).apply(PBE_TO_NASH()).apply(NORMAL_FORM()).apply(IESDS()).apply(SUPPORT_ENUM()).solve(claim=claim)


def test_c8_baseline_equilibrium_matches_round_6():
    r = solve_composite(composite())
    assert len(r.solutions) == 4 and len(r.classes) == 1       # F4: four representations, one solution under the PBE equivalence
    assert behaviour(r.solution) == (0.8889, 0.8611, 0.5652, -0.2239, 0.0)
    assert [rec.rule for rec in r.trace] == ["PBE_TO_NASH", "NORMAL_FORM", "IESDS", "SUPPORT_ENUM"]


def test_c8_expensive_decoy_matches_round_6():
    r = solve_composite(composite(cd=0.5))
    assert {behaviour(s)[:4] for s in r.solutions} == {(0.2, 0.0, 0.6758, -0.2622)}


def test_c8_leak_as_world_route_gives_multiplicity():
    r = solve_composite(composite(leak="world"))
    vals = {behaviour(s)[3] for s in r.solutions}
    assert len(vals) >= 2 and -0.2125 in vals and -0.3375 in vals
    with pytest.raises(ValueError):          # selection among equilibria is not declared: the kernel will not pick
        r.solution


def test_c8_leak_as_record_arrow_is_refused_by_the_nash_reduction():
    """A noiseless record arrow D -> B leaves some of B's records unreached after some histories, so PBE != Nash
    is no longer certifiable. The world-route leak (round 6's actual model) kept full support; the arrow does not."""
    with pytest.raises(Rejected) as e:
        Derivation(composite(leak="arrow")).apply(PBE_TO_NASH())
    assert "full-support observations" in e.value.names()


def test_c8_reject_pbe_to_nash_without_noise():
    with pytest.raises(Rejected) as e:
        Derivation(composite(eps=0.0)).apply(PBE_TO_NASH())
    assert "full-support observations" in e.value.names()


def test_c8_reject_obs_as_record_for_the_noisy_dip():
    P = composite().copy(concept="nash")
    with pytest.raises(Rejected) as e:
        Derivation(P).apply(OBS_AS_RECORD("eB1", "o1"))
    assert "noise-free, single-source observation" in e.value.names()


# ------------------------------------------------------------------ C9
def team_value(P):
    return Derivation(P).apply(TEAM_ENUM()).solve().solutions[0]["value"]


def test_c9_obs_as_record_is_exact_even_with_side_effects():
    for g in (0.0, 0.05):
        P = channel("world", gamma=g)
        r = Derivation(P).apply(OBS_AS_RECORD("eP", "o")).apply(TEAM_ENUM()).solve(claim={"sound", "complete"})
        assert abs(r.solutions[0]["value"] - team_value(P)) < 1e-12


def test_c9_world_channel_as_message_valid_when_side_conditions_hold():
    P = channel("world")
    r = Derivation(P).apply(WORLD_CHANNEL_AS_MESSAGE("eP", "o")).apply(TEAM_ENUM()).solve(claim={"sound", "complete"})
    assert abs(r.solutions[0]["value"] - 0.75) < 1e-12
    assert abs(team_value(channel("record")) - 0.75) < 1e-12        # the independently built pure-message mechanism


@pytest.mark.parametrize("kw, failed", [({"gamma": 0.05}, "separable side effect"),
                                        ({"e": 0.1}, "noise-free, single-source observation"),
                                        ({"bystander": True}, "no other observers of the world item")])
def test_c9_world_channel_as_message_rejected(kw, failed):
    with pytest.raises(Rejected) as e:
        Derivation(channel("world", **kw)).apply(WORLD_CHANNEL_AS_MESSAGE("eP", "o"))
    assert failed in e.value.names()
    if "gamma" in kw:   # and the rejection matters: the world channel and the pure message really differ
        assert abs(team_value(channel("world", **kw)) - team_value(channel("record", **kw))) > 1e-3


def test_c9_branching_derivation_and_its_certification():
    """F8 (found by derivation search): for an argmax task, a fan-out's best branch is a solution only if the branches
    cover the policy class. With a multi-record sender, coverage fails, so SOUNDNESS (not just completeness) is dropped."""
    P = route_choice()
    plan = lambda i, sub: sub.apply(TEAM_ENUM())
    r = Derivation(P).fan(FAN_OUT_ACTIONS("C", [("rec",), ("none", "push")]), plan).solve()
    assert "sound" not in r.preserves and r.solutions[0]["value"] <= team_value(P) + 1e-12
    with pytest.raises(Rejected):
        Derivation(P).fan(FAN_OUT_ACTIONS("C", [("rec",), ("none", "push")]), plan).solve(claim={"sound"})
    Q = route_choice(single_record=True)
    r2 = Derivation(Q).fan(FAN_OUT_ACTIONS("C", [("rec",), ("none", "push")]), plan).solve(claim={"sound", "complete"})
    assert abs(r2.solutions[0]["value"] - team_value(Q)) < 1e-12
    with pytest.raises(Rejected) as e:
        Derivation(P).fan(FAN_OUT_ACTIONS("C", [("rec", "none"), ("none", "push")]), plan)
    assert "blocks partition the action set" in e.value.names()
