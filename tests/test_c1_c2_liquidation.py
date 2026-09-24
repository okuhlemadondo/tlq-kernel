"""C1 finance / sinh schedule and C2 unknown-drift state reduction."""
import numpy as np
import pytest
from scipy.optimize import minimize
from cases import liquidation
from tlq.kernel import Derivation, Rejected
from tlq.liquidation import (SHIFT, ADAPT, CONST_FORECAST, OPEN_LOOP, SCALE, SOLVE_RECURRENCE, SOLVE_QP,
                             CE_MARTINGALE, objective)

CHAIN = [SHIFT, ADAPT, CONST_FORECAST, OPEN_LOOP, SCALE, SOLVE_RECURRENCE]


def oracle_path(P, mu):
    """Independent oracle: unconstrained numerical optimization of the raw deterministic objective."""
    f = lambda q: -objective(P, np.r_[P.Q0, q, 0.0], mu)
    x = minimize(f, np.linspace(P.Q0, 0, P.N + 1)[1:-1], method="BFGS", options={"gtol": 1e-10}).x
    return np.r_[P.Q0, x, 0.0]


def run(P, rules, claim=("sound", "complete")):
    d = Derivation(P)
    for r in rules:
        d.apply(r)
    return d.solve(claim=claim)


# ------------------------------------------------------------------ C1 valid derivations
def test_c1_full_chain_reproduces_sinh_schedule():
    P = liquidation()
    r = run(P, CHAIN)
    s = r.solution
    assert [rec.rule for rec in r.trace] == ["SHIFT", "ADAPT", "CONST_FORECAST", "OPEN_LOOP", "SCALE", "SOLVE_RECURRENCE"]
    assert np.allclose(s["Q"], oracle_path(P, 0.0), atol=1e-4)
    k = np.arccosh(1 + 0.3 / 2)
    assert abs(s["u"][0] - (1 - np.sinh(9 * k) / np.sinh(10 * k))) < 1e-12        # marginal allocation law
    assert abs(s["u"][0] - (1 - np.exp(-k))) < 1e-3
    assert abs(s["F"] - 820.86) < 0.01                                            # Q0 X0 - cost*
    assert set(r.ledger) == {("ADAPT", "increments integrable"), ("CONST_FORECAST", "forecast E[xi_t | F_{t-1}] is deterministic")}
    kinds = {(rec.rule, name): (kind, status) for rec in r.trace for name, kind, status, _ in rec.checked}
    assert kinds[("SHIFT", "forced liquidation (Q_N = 0)")] == ("S", "ok")
    assert kinds[("SOLVE_RECURRENCE", "no inequality constraint active")] == ("V", "ok")
    assert r.preserves == frozenset({"sound", "complete"})


def test_c1_monte_carlo_agrees_with_certified_value():
    P = liquidation()
    s = run(P, CHAIN).solution
    rng = np.random.default_rng(0)
    M, N = 100000, P.N
    xi = rng.standard_t(4, size=(M, N))            # heavy tails: irrelevant for this target
    X = P.X0 + np.c_[np.zeros(M), np.cumsum(xi, 1)][:, :N]
    a = -np.diff(s["Q"])
    J = (a * X - P.eta * a ** 2).sum(1) - P.phi * (s["Q"][1:N] ** 2).sum()
    assert abs(J.mean() - s["F"]) < 4 * J.std() / np.sqrt(M)


def test_c1_drift_skips_scale_and_matches_oracle():
    P = liquidation(context={"forecast": "constant", "mu": 5.0, "integrable": True})
    s = run(P, [SHIFT, ADAPT, CONST_FORECAST, OPEN_LOOP, SOLVE_RECURRENCE]).solution
    assert np.allclose(s["Q"], oracle_path(P, 5.0), atol=1e-4)


# ------------------------------------------------------------------ C1 invalid derivations the kernel must reject
def test_c1_reject_shift_when_liquidation_not_forced():
    with pytest.raises(Rejected) as e:
        Derivation(liquidation(terminal="free")).apply(SHIFT)
    assert "forced liquidation (Q_N = 0)" in e.value.names()


def test_c1_reject_adapt_before_shift():
    with pytest.raises(Rejected) as e:
        Derivation(liquidation()).apply(ADAPT)
    assert "applicability" in e.value.names()


def test_c1_reject_adapt_for_mean_variance_target():
    with pytest.raises(Rejected) as e:
        Derivation(liquidation(target="mean_variance")).apply(SHIFT).apply(ADAPT)
    assert "target is an expectation of the payoff" in e.value.names()


def test_c1_reject_adapt_for_anticipative_policies():
    with pytest.raises(Rejected) as e:
        Derivation(liquidation(policies="anticipative")).apply(SHIFT).apply(ADAPT)
    assert "admissible policies are adapted" in e.value.names()


def test_c1_reject_scale_with_drift():
    P = liquidation(context={"forecast": "constant", "mu": 5.0, "integrable": True})
    with pytest.raises(Rejected) as e:
        Derivation(P).apply(SHIFT).apply(ADAPT).apply(CONST_FORECAST).apply(OPEN_LOOP).apply(SCALE)
    assert "homogeneous of degree 2 (zero drift)" in e.value.names()


def test_c1_reject_closed_form_when_constraints_bind_and_route_to_qp():
    P = liquidation(context={"forecast": "constant", "mu": 100.0, "integrable": True})
    with pytest.raises(Rejected) as e:
        run(P, [SHIFT, ADAPT, CONST_FORECAST, OPEN_LOOP, SOLVE_RECURRENCE])
    assert "no inequality constraint active" in e.value.names()
    s = run(P, [SHIFT, ADAPT, CONST_FORECAST, OPEN_LOOP, SOLVE_QP]).solution
    assert np.all(np.diff(s["Q"]) <= 1e-6) and s["Q"][-1] == 0
    rng = np.random.default_rng(1)          # no feasible perturbation improves the certified solution
    for _ in range(200):
        q = np.clip(s["Q"][1:-1] + rng.normal(0, 1.0, P.N - 1), 0, P.Q0)
        q = np.minimum.accumulate(q)
        assert objective(P, np.r_[P.Q0, q, 0.0], 100.0) <= objective(P, s["Q"], 100.0) + 1e-6


def test_c1_reject_integrability_contradicted_by_world_model():
    P = liquidation(context={"forecast": "constant", "mu": 0.0, "integrable": False})
    with pytest.raises(Rejected) as e:
        Derivation(P).apply(SHIFT).apply(ADAPT)
    assert "increments integrable" in e.value.names()


# ------------------------------------------------------------------ C2 unknown drift
BAYES = {"forecast": "bayes_gaussian", "m0": 0.0, "s0": 1.0, "sigma": 1.0, "integrable": True}


def dp_oracle_action(P, t, Q, m):
    """Independent oracle: exact backward DP with quadratic value V_s(Q,m) = -A_s Q^2 + B_s Q m + c(m)."""
    A = np.zeros(P.N); B = np.zeros(P.N)
    A[P.N - 1], B[P.N - 1] = P.eta, 0.0
    for s in range(P.N - 2, -1, -1):
        d = P.eta + P.phi + A[s + 1]
        A[s] = P.eta - P.eta ** 2 / d
        B[s] = P.eta * (1 + B[s + 1]) / d
    if t == P.N - 1:
        return Q
    d = P.eta + P.phi + A[t + 1]
    return Q - (2 * P.eta * Q + (1 + B[t + 1]) * m) / (2 * d)


def test_c2_unknown_drift_reduces_to_certainty_equivalent_controller():
    P = liquidation(context=BAYES)
    r = Derivation(P).apply(SHIFT).apply(ADAPT).apply(CE_MARTINGALE).solve(claim={"sound", "complete"})
    s = r.solution
    assert s["state"] == ("t", "Q", "X_t - X_0")                 # the derived information state
    ctrl, mean = s["controller"], s["posterior_mean"]
    for t, Q, disp in [(0, 100.0, 0.0), (3, 60.0, 2.5), (5, 40.0, -3.0), (8, 10.0, 1.0)]:
        assert abs(ctrl(t, Q, disp) - dp_oracle_action(P, t, Q, mean(t, disp))) < 1e-10
    assert ("CE_MARTINGALE", "no inequality constraint binds on realized paths") in r.ledger


def test_c2_reject_const_forecast_when_world_model_has_unknown_drift():
    with pytest.raises(Rejected) as e:
        Derivation(liquidation(context=BAYES)).apply(SHIFT).apply(ADAPT).apply(CONST_FORECAST)
    assert "forecast E[xi_t | F_{t-1}] is deterministic" in e.value.names()


def test_c2_reject_ce_without_a_prior():
    with pytest.raises(Rejected) as e:
        Derivation(liquidation()).apply(SHIFT).apply(ADAPT).apply(CE_MARTINGALE)
    assert "Bayesian model with a specified prior" in e.value.names()
