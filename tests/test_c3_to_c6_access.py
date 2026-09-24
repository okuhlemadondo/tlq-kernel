"""C3 13/20 information structure, C4 driver (4, 4/3, 1), C5 silence vs absence, C6 endogenous absence."""
import pytest
from cases import parent_child, driver, silence, spawning, cycle
from tlq.kernel import Derivation, Rejected
from tlq.rules_finite import TEAM_ENUM, BEHAVIOURAL_OPT, COMMON_INFO, COORDINATOR_DP, NULL_AS_ABSENCE


def value(P, rules, claim=("sound",)):
    d = Derivation(P)
    for r in rules:
        d.apply(r)
    return d.solve(claim=claim)


# ------------------------------------------------------------------ C3
def test_c3_brute_force_and_common_information_agree_on_13_20():
    P = parent_child()
    r1 = value(P, [TEAM_ENUM()], claim={"sound", "complete"})
    r2 = value(P, [COMMON_INFO(), COORDINATOR_DP()], claim={"sound", "complete"})
    assert abs(r1.solutions[0]["value"] - 0.65) < 1e-12
    assert abs(r2.solution["value"] - 0.65) < 1e-12
    assert abs(P.value(r2.solution["policy"])["team"] - 0.65) < 1e-12      # lifted distributed controller executes
    assert [rec.rule for rec in r2.trace] == ["COMMON_INFO", "COORDINATOR_DP"]


def test_c3_interpretation_depends_on_the_other_sites_policy():
    P = parent_child()
    beliefs = []
    for child in [{(("y", 0),): 0, (("y", 1),): 1}, {(("y", 0),): 1, (("y", 1),): 1}, {(("y", 0),): 1, (("y", 1),): 0}]:
        num = den = 0.0
        for pw, om in P.world:
            for p, acts, _, _ in P.paths({"child": child, "parent": "uniform"}, om):
                if acts["e_child"] == 1:
                    den += pw * p; num += pw * p * (om["b"] == 1)
        beliefs.append(round(num / den, 12))
    assert beliefs == [0.75, 0.5, 0.25]


def test_c3_reading_the_child_record_is_worth_3_4():
    assert abs(value(parent_child(parent_reads_child_record=True), [TEAM_ENUM()]).solutions[0]["value"] - 0.75) < 1e-12


def test_c3_reject_common_information_without_partial_history_sharing():
    with pytest.raises(Rejected) as e:
        Derivation(parent_child(auditor=True)).apply(COMMON_INFO())
    assert "partial history sharing" in e.value.names()


def test_c3_reject_overclaim_full():
    with pytest.raises(Rejected) as e:
        value(parent_child(), [TEAM_ENUM()], claim={"full"})
    assert "claim" in e.value.names()


# ------------------------------------------------------------------ C4
def test_c4_forgetful_fresh_coin_is_4_3():
    r = value(driver(memory=False, coin="fresh"), [BEHAVIOURAL_OPT()], claim={"sound", "complete"})
    s = r.solution
    assert abs(s["value"] - 4 / 3) < 1e-6
    (rec, dist), = s["policy"]["driver"].items()
    assert abs(dist["continue"] - 2 / 3) < 1e-4


def test_c4_forgetful_persistent_coin_is_1():
    r = value(driver(memory=False, coin="persistent"), [TEAM_ENUM()], claim={"sound", "complete"})
    assert abs(r.solutions[0]["value"] - 1.0) < 1e-12


def test_c4_perfect_recall_is_4():
    r = value(driver(memory=True, coin="fresh"), [TEAM_ENUM()], claim={"sound", "complete"})
    assert abs(r.solutions[0]["value"] - 4.0) < 1e-12


def test_c4_reject_pure_enumeration_for_forgetful_fresh_driver():
    with pytest.raises(Rejected) as e:
        Derivation(driver(memory=False, coin="fresh")).apply(TEAM_ENUM())
    assert "pure plans are value-complete despite fresh randomness" in e.value.names()


def test_c4_reject_behavioural_strategies_without_fresh_randomness():
    with pytest.raises(Rejected) as e:
        Derivation(driver(memory=False, coin=None)).apply(BEHAVIOURAL_OPT())
    assert "behavioural strategies admissible (fresh randomness at every decision site)" in e.value.names()


def test_c4_semantic_regression_sites_are_not_evaluations():
    """Encoding the two evaluations as two sites silently grants perfect recall: value 4, not 4/3."""
    r = value(driver(memory=False, coin="fresh", two_sites=True), [TEAM_ENUM()], claim={"sound", "complete"})
    assert abs(r.solutions[0]["value"] - 4.0) < 1e-12


# ------------------------------------------------------------------ C5
def test_c5_silence_is_a_value_absence_is_no_arrow():
    with_silence = value(silence(), [TEAM_ENUM()]).solutions[0]["value"]
    no_silence = value(silence(silence_allowed=False), [TEAM_ENUM()]).solutions[0]["value"]
    no_channel = value(silence(channel=False), [TEAM_ENUM()]).solutions[0]["value"]
    assert abs(with_silence - 0.75) < 1e-12 and abs(no_silence - 0.70) < 1e-12 and abs(no_channel - 0.5) < 1e-12


def test_c5_reject_treating_silence_as_uninformative():
    with pytest.raises(Rejected) as e:
        Derivation(silence()).apply(NULL_AS_ABSENCE("parent", "m", "S", uninformed_action=0))
    assert "null reading is uninformative under every admissible program" in e.value.names()


# ------------------------------------------------------------------ C6
def test_c6_endogenous_absence_value_3_4():
    assert abs(value(spawning(), [TEAM_ENUM()]).solutions[0]["value"] - 0.75) < 1e-12


def test_c6_reject_absence_as_uninformative_when_spawning_is_endogenous():
    with pytest.raises(Rejected) as e:
        Derivation(spawning()).apply(NULL_AS_ABSENCE("R", "c", "none", uninformed_action=1))
    assert "null reading is uninformative under every admissible program" in e.value.names()


def test_c6_what_the_uncertified_step_would_cost():
    """Asymmetric prior removes tie artifacts: certified 0.7 vs uncertified (bypassing the kernel) 0.6."""
    P = spawning(prior=0.6)
    certified = value(P, [TEAM_ENUM()]).solutions[0]["value"]
    bypass = NULL_AS_ABSENCE("R", "c", "none", uninformed_action=1).forward(P)    # what a non-certifying system does
    uncertified = TEAM_ENUM().forward(bypass).solutions[0]["value"]
    assert abs(certified - 0.7) < 1e-12 and abs(uncertified - 0.6) < 1e-12


def test_c6_absence_accepted_when_it_is_uninformative():
    P = spawning(endogenous=False)
    r = value(P, [NULL_AS_ABSENCE("R", "c", "none", uninformed_action=0), TEAM_ENUM()], claim={"sound", "complete"})
    assert abs(r.solutions[0]["value"] - value(P, [TEAM_ENUM()]).solutions[0]["value"]) < 1e-12


# ------------------------------------------------------------------ execution semantics
def test_cycle_rejected_by_execution_semantics():
    with pytest.raises(Rejected) as e:
        Derivation(cycle()).apply(TEAM_ENUM())
    assert "execution semantics well-founded" in e.value.names()
