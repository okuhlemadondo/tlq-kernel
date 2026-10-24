"""Tests for the S/L/D/V Kernel Obligation Spine & Named Rejections on Vertical Rewrites."""
from __future__ import annotations
import pytest
from tlq.kernel import Derivation, Rejected
from tlq.vertical import (
    ResizingProblem,
    MTM_SHIFT,
    ADAPT_INCREMENTS,
    CUMULANT_QUAD,
    HAMILTONIAN_COMPLETE,
    SOLVE_KELLY_CLAMPED,
    VERTICAL_CHAIN,
)


def _run_chain(problem: ResizingProblem, rules=VERTICAL_CHAIN, claim=("sound", "complete")):
    d = Derivation(problem)
    for r in rules:
        d.apply(r)
    return d.solve(claim=claim)


def test_valid_vertical_derivation_chain_and_ledger_tracking():
    """Verifies the 3-stage vertical derivation chain reproduces clamped Kelly target and records D obligations."""
    p = ResizingProblem(
        W0=100.0,
        Q0=5.0,
        P0=50.0,
        f_bar=0.15,
        sigma2_bar=0.25,
        sigma_min2=0.05,
        xi_max=0.5,
        eta_margin=0.2,
        third_abs_moment=0.06,
    )
    res = _run_chain(p, claim=("sound", "complete", "full"))
    sol = res.solution
    assert [rec.rule for rec in res.trace] == [
        "MTM_SHIFT",
        "ADAPT_INCREMENTS",
        "CUMULANT_QUAD",
        "HAMILTONIAN_COMPLETE",
        "SOLVE_KELLY_CLAMPED",
    ]
    assert abs(sol["theta_star"] - 0.60) <= 1e-12
    assert abs(sol["Q_next_star"] - 60.0) <= 1e-12
    assert abs(sol["potential_offset"] - 250.0) <= 1e-12
    assert res.total_slack > 0.0
    assert set(res.ledger) == {
        ("ADAPT_INCREMENTS", "shocks have finite third absolute moment"),
        ("CUMULANT_QUAD", "bounded shock support |xi_{t+1}| <= xi_max a.s. (Definition 2.3)"),
    }


def test_reject_mtm_shift_without_mark_to_market_terminal_potential():
    with pytest.raises(Rejected) as exc:
        Derivation(ResizingProblem(terminal_potential="none")).apply(MTM_SHIFT)
    assert "mark-to-market terminal potential (Psi(W_N, Q_N) = Q_N P_N)" in exc.value.names()


def test_reject_adapt_when_policies_are_anticipative():
    with pytest.raises(Rejected) as exc:
        Derivation(ResizingProblem(policies="anticipative")).apply(MTM_SHIFT).apply(ADAPT_INCREMENTS)
    assert "admissible policies are F_t-adapted (Axiom 1)" in exc.value.names()


def test_reject_adapt_when_integrability_contradicted():
    with pytest.raises(Rejected) as exc:
        Derivation(ResizingProblem(integrable=False)).apply(MTM_SHIFT).apply(ADAPT_INCREMENTS)
    assert "shocks have finite third absolute moment" in exc.value.names()


def test_reject_cumulant_quad_when_variance_below_floor():
    p = ResizingProblem(sigma2_bar=0.02, sigma_min2=0.05)
    with pytest.raises(Rejected) as exc:
        Derivation(p).apply(MTM_SHIFT).apply(ADAPT_INCREMENTS).apply(CUMULANT_QUAD)
    assert "positive conditional variance floor (sigma2_bar >= sigma_min2 > 0)" in exc.value.names()


def test_reject_claiming_full_zero_slack_preservation_on_log_wealth_without_surrogate_slack():
    """Theorem 2.4(1) counterexample refutes exact zero-slack argmax equality for log_wealth."""
    p = ResizingProblem(target="log_wealth", allow_surrogate_slack=False)
    with pytest.raises(Rejected) as exc:
        _run_chain(p, claim=("sound", "complete", "full"))
    assert "claim" in exc.value.names()

    # Linear wealth requires zero surrogate slack and preserves 'full'
    p_lin = ResizingProblem(target="linear_wealth", allow_surrogate_slack=False)
    res_lin = _run_chain(p_lin, claim=("sound", "complete", "full"))
    assert res_lin.total_slack == 0.0
